"""Side-view (x-z) filmstrip of a grasp trial, drawn from the REAL simulated
geom positions (no OpenGL needed). Verifies the gripper visibly descends,
closes on the object, and lifts it (or, for a failure case, does not)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import mujoco
import contact_probe as cp
import m2_floating_gripper_grasp as M


def capture(params, snaps=6):
    m = mujoco.MjModel.from_xml_string(
        M.scene_xml(params["d"], params["h"], params["mass"], params["mu"], params["force"]))
    cp.force_dense_jacobian(m)
    d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
         for n in ("apx", "apy", "apz", "alf", "arf")}
    gid = {g: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
           for g in ("lfinger", "rfinger", "object", "pedestal", "palm")}
    obj_z = 0.20 + params["h"] / 2 + 0.001
    gx = params.get("xoff", 0.0)
    grasp_z = obj_z + params.get("zoff", 0.0) + 0.065
    hover_z = grasp_z + 0.15; lift_z = grasp_z + 0.15

    def C(pz, lf, rf):
        d.ctrl[A["apx"]] = gx; d.ctrl[A["apy"]] = 0; d.ctrl[A["apz"]] = pz - M.PALM_HOME
        d.ctrl[A["alf"]] = lf; d.ctrl[A["arf"]] = rf

    frames, labels = [], []
    plan = ([("hover", hover_z, hover_z, 0.0, 0.0, 60)]
            + [("descend", hover_z, grasp_z, 0.0, 0.0, 80)]
            + [("close", grasp_z, grasp_z, 0.03, 0.03, 60)]
            + [("hold", grasp_z, grasp_z, 0.03, 0.03, 40)]
            + [("lift", grasp_z, lift_z, 0.03, 0.03, 90)]
            + [("held", lift_z, lift_z, 0.03, 0.03, 60)])
    log = []
    for name, z0, z1, lf, rf, n in plan:
        for k in range(n):
            a = (k + 1) / n
            C(z0 + a * (z1 - z0), lf, rf); mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        snap = {g: (np.array(d.geom_xpos[gid[g]]).copy(), np.array(m.geom_size[gid[g]]).copy())
                for g in gid}
        snap["h"] = params["h"]
        log.append((name, snap, float(d.geom_xpos[gid["object"]][2])))
    return log


def draw(ax, name, snap, oz0):
    def rect(g, color, fill=True, hw=None, hh=None):
        p, s = snap[g]
        w = (hw if hw is not None else s[0]); h = (hh if hh is not None else s[2])
        ax.add_patch(Rectangle((p[0] - w, p[2] - h), 2 * w, 2 * h,
                               facecolor=color if fill else "none",
                               edgecolor=color, lw=1.4, alpha=0.85))
    # pedestal, palm, fingers, object (cylinder side view = rectangle of half-height h/2)
    rect("pedestal", "#b0b0b0")
    rect("palm", "#7a7a7a")
    rect("lfinger", "#c34317"); rect("rfinger", "#c34317")
    rect("object", "#2f6f9f", hh=snap["h"] / 2)
    ax.axhline(0.20, color="#888", lw=0.6, ls="--")           # pedestal top
    ax.set_xlim(-0.13, 0.13); ax.set_ylim(0.15, 0.62)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s\nobjZ=%.3f" % (name, snap["object"][0][2]), fontsize=9)


def filmstrip(params, path, title):
    log = capture(params)
    fig, axes = plt.subplots(1, len(log), figsize=(2.05 * len(log), 3.4))
    for ax, (name, snap, oz) in zip(axes, log):
        draw(ax, name, snap, log[0][2])
    rise = (log[-1][2] - log[0][2]) * 1000
    fig.suptitle("%s   (object rise: %+.0f mm)" % (title, rise), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path, " rise %+.0f mm" % rise)


if __name__ == "__main__":
    base = dict(M.BASE)
    filmstrip(base, "film_success.png", "Floating-gripper grasp — SUCCESS (base)")
    slip = dict(M.BASE); slip.update(dict(mu=0.15))
    filmstrip(slip, "film_slip.png", "Low friction (mu=0.15) — SLIP / no lift")
    miss = dict(M.BASE); miss.update(dict(xoff=0.03))
    filmstrip(miss, "film_miss.png", "Large lateral offset (xoff=30mm) — MISS")
