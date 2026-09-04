"""
z_local_schema.py  —  the FROZEN per-contact local feature schema for the
Factorized Interaction World Model.

Design contract (this is the whole point of the factorization):
  * z_local must be EMBODIMENT-AGNOSTIC and expressed in contact/object frames,
    never in robot joint space or world pose. If two different robots create the
    same local contact situation, they must produce the same z_local — that is
    what lets a frozen C_theta transfer across embodiments.
  * z_local is what the learned local constitutive law consumes:
        eta_i = C_theta(z_local_i)
    and is composed with the ANALYTICAL global coupling W = J_c M^-1 J_c^T by the
    global solve  lambda = S(W, phi, phidot, {eta_i}).  Supervision is on the
    OBSERVABLE object motion, never on solver-internal impulses (those are logged
    only as diagnostics).

  * The self-inertia block W_ii is logged SEPARATELY from the strict-local
    vector, so RQ1 (strict-local vs W_ii-aware) can be tested by ablation WITHOUT
    regenerating data. Freeze the schema now; scale later.

Everything is computed from the MuJoCo contact + the validated contact_probe.
Curvatures are exact for primitive geoms (plane/box/sphere/cylinder/capsule) and
NaN for meshes (fill with a mesh estimator later, without a re-run).

SCHEMA_VERSION bump => regenerate. Keep it stable.
"""

import numpy as np
import mujoco
import contact_probe as cp

SCHEMA_VERSION = "z_local.v1.1"

# Strict-local feature layout (order is frozen; v1.1 APPENDS slip_accum+age so the
# first 25 dims are byte-identical to v1). dims in comments.
STRICT_FIELDS = [
    ("gap",            1),   # signed distance phi (m); <0 = penetration
    ("cpos_obj",       3),   # contact point in OBJECT frame (m) — where on the object
    ("normal_obj",     3),   # contact normal in OBJECT frame (unit) — orientation on the object
    ("vn",             1),   # normal relative velocity phidot (m/s); >0 separating
    ("vt",             2),   # tangential relative velocity in contact frame (m/s) — slip
    ("wn",             1),   # relative angular velocity about the normal (rad/s) — spin
    ("kappa_obj",      2),   # object-side principal curvatures (1/m)
    ("kappa_other",    2),   # other-geom principal curvatures (1/m)
    ("mu",             3),   # friction: (tangential, torsional, rolling)
    ("solref",         2),   # normal compliance (time-const, damping-ratio)
    ("solimp",         5),   # constraint impedance profile
    ("slip_accum",     1),   # v1.1: integral of |v_t| dt since contact onset (m)
    ("contact_age",    1),   # v1.1: time since this contact first appeared (s)
]
STRICT_DIM = sum(d for _, d in STRICT_FIELDS)   # = 27

# Logged alongside (NOT part of strict z_local): the local self-inertia block.
AUX_FIELDS = [
    ("W_ii", 9),            # 3x3 self-block of the Delassus operator, row-major
]

# Per-step object-level TARGET (shared by all contacts in that step).
TARGET_FIELDS = [
    ("dp", 3), ("drot", 3), ("obj_v", 3), ("obj_w", 3), ("dt", 1),
]

# Diagnostics only (never supervised).
DIAG_FIELDS = [
    ("lambda", 3),          # resolved contact impulse/force on the efc rows
    ("Fn", 1),
]


# ----------------------------------------------------------------------
# primitive curvature (principal curvatures at the contact, 1/m)
# ----------------------------------------------------------------------
def _principal_curvatures(model, geom_id, normal_world, geom_xmat):
    """Approximate principal curvatures of a primitive geom at a contact whose
    outward normal is `normal_world`. Returns (k1, k2). NaN for meshes."""
    gtype = int(model.geom_type[geom_id])
    s = model.geom_size[geom_id]
    T = mujoco.mjtGeom
    if gtype == T.mjGEOM_PLANE:
        return 0.0, 0.0
    if gtype == T.mjGEOM_BOX:
        return 0.0, 0.0                        # face contact (approx); edges handled later
    if gtype == T.mjGEOM_SPHERE:
        return 1.0 / s[0], 1.0 / s[0]
    if gtype in (T.mjGEOM_CYLINDER, T.mjGEOM_CAPSULE):
        r = s[0]
        axis = np.array(geom_xmat).reshape(3, 3)[:, 2]     # local z in world
        cosang = abs(float(np.dot(np.asarray(normal_world), axis)))
        if cosang > 0.85:                       # contact on the flat cap / end
            return (0.0, 0.0) if gtype == T.mjGEOM_CYLINDER else (1.0 / r, 1.0 / r)
        return 1.0 / r, 0.0                     # on the side: circumferential 1/r, axial 0
    if gtype == T.mjGEOM_ELLIPSOID:
        return float("nan"), float("nan")
    return float("nan"), float("nan")           # mesh/hfield: fill later


# ----------------------------------------------------------------------
# the feature extractor
# ----------------------------------------------------------------------
class ContactTracker:
    """Persists contacts across steps (MuJoCo gives no persistent id) so we can
    accumulate slip and age. Matches current->previous by (geom pair) + nearest
    world point within `match_dist`. Call update(model,data,dt) ONCE per step;
    it returns {contact_index: (slip_accum, contact_age)} for that step."""
    def __init__(self, match_dist=0.006):
        self.match_dist = match_dist
        self.tracks = {}          # tid -> dict(pos, slip, age, pair)
        self._next = 0

    def _tangential_speed(self, model, data, ci):
        c = data.contact[ci]; addr, dim = int(c.efc_address), int(c.dim)
        J = cp.contact_jacobian(data, model.nv)
        if J.shape[0] == 0 or dim < 3:
            return 0.0
        vc = J[addr:addr + dim] @ np.array(data.qvel)
        return float(np.hypot(vc[1], vc[2]))

    def update(self, model, data, dt):
        out = {}
        alive = set()
        for ci in range(int(data.ncon)):
            c = data.contact[ci]
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            pos = np.array(c.pos)
            vt = self._tangential_speed(model, data, ci)
            # find nearest existing track with the same geom pair
            best, bestd = None, self.match_dist
            for tid, tr in self.tracks.items():
                if tr["pair"] != pair or tid in alive:
                    continue
                d = float(np.linalg.norm(tr["pos"] - pos))
                if d < bestd:
                    best, bestd = tid, d
            if best is None:
                best = self._next; self._next += 1
                self.tracks[best] = dict(pos=pos, slip=0.0, age=0.0, pair=pair)
            tr = self.tracks[best]
            tr["slip"] += vt * dt; tr["age"] += dt; tr["pos"] = pos
            alive.add(best)
            out[ci] = (tr["slip"], tr["age"])
        # expire tracks not seen this step
        for tid in [t for t in self.tracks if t not in alive]:
            del self.tracks[tid]
        return out


def contact_features(model, data, ci, obj_body_id, obj_geom_id, W=None, dt=None,
                     slip_accum=0.0, contact_age=0.0):
    """Full record for one contact: strict z_local + W_ii + diagnostics.
    `W` (full Delassus, from cp.assemble_W) optional; if given, W_ii is filled."""
    c = data.contact[ci]
    frame = np.array(c.frame).reshape(3, 3)             # rows: normal, t1, t2
    n_w, t1_w, t2_w = frame[0], frame[1], frame[2]
    addr, dim = int(c.efc_address), int(c.dim)

    # object pose (to express things in the object frame)
    R_obj = np.array(data.xmat[obj_body_id]).reshape(3, 3)
    p_obj = np.array(data.xpos[obj_body_id])
    cpos_obj = R_obj.T @ (np.array(c.pos) - p_obj)
    normal_obj = R_obj.T @ n_w

    # relative contact-frame velocity = (efc_J @ qvel) on this contact's rows
    J = cp.contact_jacobian(data, model.nv)
    vc = (J[addr:addr + dim] @ np.array(data.qvel)) if J.shape[0] else np.zeros(dim)
    vn = float(vc[0]); vt = np.array([vc[1] if dim > 1 else 0.0,
                                      vc[2] if dim > 2 else 0.0])

    # spin about the normal from the two bodies' angular velocities
    def body_ang(bid):
        v6 = np.zeros(6); mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
        return v6[:3]
    b1 = model.geom_bodyid[c.geom1]; b2 = model.geom_bodyid[c.geom2]
    wn = float(np.dot(body_ang(b2) - body_ang(b1), n_w))

    # which geom is the object vs the other
    other_geom = c.geom1 if c.geom2 == obj_geom_id else c.geom2
    k_obj = _principal_curvatures(model, obj_geom_id, n_w, data.geom_xmat[obj_geom_id]) \
        if (c.geom1 == obj_geom_id or c.geom2 == obj_geom_id) else (float("nan"), float("nan"))
    k_other = _principal_curvatures(model, other_geom, -np.array(n_w), data.geom_xmat[other_geom])

    mu = np.array(c.friction)                          # (tangential,tangential,spin,roll,roll)
    strict = dict(
        gap=np.array([float(c.dist)]),
        cpos_obj=cpos_obj, normal_obj=normal_obj,
        vn=np.array([vn]), vt=vt, wn=np.array([wn]),
        kappa_obj=np.array(k_obj), kappa_other=np.array(k_other),
        mu=np.array([mu[0], mu[2], mu[3]]),
        solref=np.array(c.solref)[:2], solimp=np.array(c.solimp)[:5],
        slip_accum=np.array([float(slip_accum)]),
        contact_age=np.array([float(contact_age)]),
    )
    # aux: self-inertia block W_ii
    if W is not None and W.shape[0] >= addr + dim:
        Wii = W[addr:addr + dim, addr:addr + dim]
        Wii9 = np.zeros((3, 3)); Wii9[:dim, :dim] = Wii
    else:
        Wii9 = np.full((3, 3), np.nan)
    # diagnostics: resolved impulse on the rows
    ef = np.array(data.efc_force)
    lam = ef[addr:addr + dim] if ef.size >= addr + dim else np.zeros(dim)
    lam3 = np.zeros(3); lam3[:len(lam)] = lam
    aux = dict(W_ii=Wii9.reshape(-1))
    diag = dict(**{"lambda": lam3}, Fn=np.array([float(lam[0]) if len(lam) else 0.0]))
    return dict(strict=strict, aux=aux, diag=diag,
                obj_geom=int(obj_geom_id), other_geom=int(other_geom),
                efc_rows=list(range(addr, addr + dim)), dim=dim)


def strict_vector(feat):
    """Flatten the strict-local dict into the frozen-order vector (STRICT_DIM,)."""
    s = feat["strict"]
    return np.concatenate([np.atleast_1d(s[name]).astype(np.float64) for name, _ in STRICT_FIELDS])


def describe():
    print("SCHEMA", SCHEMA_VERSION, " strict_dim =", STRICT_DIM)
    off = 0
    for name, d in STRICT_FIELDS:
        print("  [%2d:%2d] %-12s %dD" % (off, off + d, name, d)); off += d
    print("  aux:", AUX_FIELDS, " target:", TARGET_FIELDS, " diag:", DIAG_FIELDS)
