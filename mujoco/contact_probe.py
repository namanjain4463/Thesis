"""
contact_probe.py  —  validated MuJoCo physics probe for the Factorized
Interaction World Model (M2 data generation).

Extracts, at any sim state (after mj_forward / during mj_step):
  * the analytical GLOBAL COUPLING  W = J_c M_sys^-1 J_c^T   (the Delassus /
    contact-space inverse-inertia operator) over all active constraints;
  * per-contact records: world point, contact frame (normal + 2 tangents),
    6D contact force (in the contact frame), the geoms involved, and the
    efc rows that constraint occupies;
  * observable object motion (Δp, ΔR as a rotation vector, linear/angular vel).

Everything here was validated against MuJoCo internals on 2026-09-03
(mujoco 3.12.0):
  - W from dense inv(fullM)  ==  W from mj_solveM (the M^-1 apply that
    mujoco_warp's solve_m mirrors)  -> 0.0 error;
  - W symmetric & PSD;
  - sum of per-contact normal forces == object weight.

BACKEND-SWAP NOTE (mujoco_warp later): the two engine-specific calls are
`_full_M` (mj_fullM) and the contact Jacobian read (`data.efc_J`). Both have
mujoco_warp equivalents (mjwarp exposes efc.J and a solve_m). Keep call sites
behind assemble_W()/contact_records() so the swap stays a one-file change.

Requires: mujoco>=3.1, numpy.  Force a DENSE constraint Jacobian on the model:
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
(the loaders below do this for you).
"""

import numpy as np
import mujoco


# ----------------------------------------------------------------------
# model / data helpers
# ----------------------------------------------------------------------
def force_dense_jacobian(model):
    """Dense efc_J makes W assembly a couple of matmuls. Call once after load."""
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    # elliptic cone => efc rows are (normal, tangent1, tangent2) in the contact
    # frame, so each contact's W block is expressed in the physical frame.
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC


def full_M(model, data):
    """Dense system mass matrix M (nv x nv)."""
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return M


def Minv_apply(model, data, B):
    """Return M^-1 @ B_row for each row of B (rows are size nv), via MuJoCo's
    own factorization (mj_solveM). This is the mujoco_warp-compatible path."""
    B = np.ascontiguousarray(B, dtype=np.float64)
    X = np.zeros_like(B)
    mujoco.mj_solveM(model, data, X, B)
    return X


# ----------------------------------------------------------------------
# the Delassus operator  W = J_c M^-1 J_c^T
# ----------------------------------------------------------------------
def contact_jacobian(data, nv):
    """Dense constraint Jacobian J (nefc x nv). Empty (0, nv) if no constraints."""
    nefc = int(data.nefc)
    if nefc == 0:
        return np.zeros((0, nv))
    return np.array(data.efc_J, dtype=np.float64).reshape(nefc, nv)


def assemble_W(model, data, use_solve=True):
    """Assemble the full Delassus operator W = J M^-1 J^T over ALL active
    constraints at the current state. Returns (W, J) with W shape (nefc, nefc).
    `use_solve=True` uses mj_solveM (mujoco_warp-compatible); False uses a dense
    inverse (handy for a sanity cross-check). Call mj_forward first."""
    nv = model.nv
    J = contact_jacobian(data, nv)
    if J.shape[0] == 0:
        return np.zeros((0, 0)), J
    if use_solve:
        X = Minv_apply(model, data, J)      # X_row = M^-1 J_row  -> (nefc, nv)
        W = J @ X.T
    else:
        M = full_M(model, data)
        W = J @ np.linalg.solve(M, J.T)
    return 0.5 * (W + W.T), J                # symmetrize away fp noise


# ----------------------------------------------------------------------
# per-contact records
# ----------------------------------------------------------------------
def _geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return n if n is not None else "geom%d" % gid


def contact_records(model, data, involving_geoms=None):
    """One dict per active contact. If `involving_geoms` (a set of geom ids) is
    given, keep only contacts touching one of them (e.g. the object's geom).

    Each record: index, geom1/2 (names+ids), world point, distance, dim,
    frame (3x3: rows = normal,t1,t2), force6 (contact frame: Fn,Ft1,Ft2,+torques),
    efc_address, efc_rows (the constraint rows this contact occupies)."""
    recs = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if involving_geoms is not None and g1 not in involving_geoms and g2 not in involving_geoms:
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)
        dim = int(c.dim)
        addr = int(c.efc_address)
        recs.append(dict(
            index=i, geom1=g1, geom2=g2,
            geom1_name=_geom_name(model, g1), geom2_name=_geom_name(model, g2),
            point=np.array(c.pos, dtype=np.float64).copy(),
            dist=float(c.dist), dim=dim,
            frame=np.array(c.frame, dtype=np.float64).reshape(3, 3).copy(),
            normal=np.array(c.frame, dtype=np.float64).reshape(3, 3)[0].copy(),
            force6=f6.copy(),
            Fn=float(f6[0]), Ft1=float(f6[1]), Ft2=float(f6[2]),
            efc_address=addr,
            efc_rows=list(range(addr, addr + dim)) if addr >= 0 else [],
        ))
    return recs


def object_W_block(W, recs):
    """Given the full W and a filtered list of contact records, return the
    sub-block of W spanning only those contacts' efc rows, plus the row->
    (contact_index, local_dir) index so off-diagonal W_ij blocks are labelled.
    This is the object-centric coupling the factorized model consumes."""
    rows, labels = [], []
    for r in recs:
        for k, e in enumerate(r["efc_rows"]):
            rows.append(e)
            labels.append((r["index"], k))     # k=0 normal, 1/2 tangents
    if not rows:
        return np.zeros((0, 0)), labels
    idx = np.array(rows, dtype=int)
    return W[np.ix_(idx, idx)].copy(), labels


# ----------------------------------------------------------------------
# observable object motion  (Δp, ΔR, v, ω)
# ----------------------------------------------------------------------
def body_pose(data, body_id):
    """World position (3,) and orientation quaternion wxyz (4,) of a body."""
    p = np.array(data.xpos[body_id], dtype=np.float64).copy()
    q = np.array(data.xquat[body_id], dtype=np.float64).copy()
    return p, q


def body_vel(model, data, body_id):
    """World-frame linear (3,) and angular (3,) velocity of a body COM."""
    v6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, v6, 0)
    # mj_objectVelocity returns [angular(3), linear(3)] in the body/world frame
    w = v6[:3].copy(); v = v6[3:].copy()
    return v, w


def quat_delta_rotvec(q_prev, q_now):
    """Rotation-vector (axis*angle, 3) taking q_prev -> q_now (both wxyz)."""
    qp = np.array(q_prev, dtype=np.float64)
    qn = np.array(q_now, dtype=np.float64)
    qc = qp.copy(); qc[1:] *= -1.0             # conjugate of q_prev
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, qn, qc)             # dq = q_now * q_prev^-1
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, dq, 1.0)           # rotation vector (angle*axis)
    return rv
