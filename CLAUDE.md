# CLAUDE.md — Thesis: Factorized Interaction World Model

PhD thesis inventing **new, certified math** for embodiment-invariant, sensing-aware
contact/manipulation transfer. Science core is in `mujoco/` (MuJoCo 3.12.0). Isaac Sim
harness in `isaac/`. Headline claim: a frozen local contact law `C_θ(z_local)` composed
with each embodiment's analytical port `Y_G` via a convex contact solve; transfer is
**certified** (covering law + friction-cone graspability + port-ID `ε_Y` + contact-law `ε_C`).

## Hard rules (never violate)
- **Only touch the `namanjain4463/Thesis` repo.** Never read, write, or push any other repo.
- Develop on branch `claude/lynxmotion-cube-picker-controller-9umls9`. **Commit + push as work lands** — the remote container is ephemeral; unpushed work is lost. (Only committed files load in the next session.)
- Isaac task: **cylinder mass ≤ 0.1 kg**. Never exceed.

## How I want you to work (stated preferences)
- **First principles. Invent, don't execute.** When I say innovate, reframe the problem and question standard assumptions — we are inventing original math, not applying known results.
- **Spend most effort thinking, validating, and self-correcting** — iterative testing, edge cases, worked examples. Find the end-to-end problems before I do.
- **Never spin.** Report weak/negative results honestly — but a negative result is a *measurement-bug suspect first*: check the probe/metric before declaring a failure.
- **Answer my direct question first**, before fixes or tangents.
- **Verify end-to-end including the VISUAL output** (renders/filmstrips), not just logs. Don't claim "done"/"works" without running it and, when relevant, looking at it.
- **Bias to building + robust testing** once a direction is set.
- **Don't over-fit to one test case.** Test diverse shapes/materials/embodiments. Establish a *predictive rule (what works and why)* before broad testing.
- **The math must generalize to bimanual + humanoids** — keep checking that, not just single arms.
- **Critically evaluate external docs** (e.g. ChatGPT drafts) — find what's wrong or overstated; don't agree by default.
- **Always give the conclusion and where it's heading**, not just the number.
- **Minimal, targeted changes** when fixing.

## Gotchas learned the hard way (don't repeat)
- A **saturating probe field fakes a failed scaling law** — use non-saturating (linear) fields to test slope-vs-L. Don't regress a solve-output (contact force) on local kinematics and call low R² a failure; pick the right observable.
- **Don't gate on structurally-zero quantities** — a 2-finger grasp is never full 6-DOF force closure, so `σ_min` of its wrench matrix is ~0 by design.
- **"Free space" means `nefc == 0` — assert it.** Object-on-pedestal contacts, joint limits, and the Panda finger **tendon** all silently contaminate port identification.
- **Panda finger tendon is a real coupling:** naive `J M⁻¹Jᵀ` is ~94% wrong for its endpoint port. *Identify `Y`, don't assume it.*
- The covering-law constant is the **sampling Lebesgue constant (≈1.4), not 1**.
- Separate "the scientific claim" from "re-deriving simulator internals" — don't rabbit-hole on MuJoCo's exact pyramidal-cone KKT when the claim only needs the frictionless case + the P1 identity.

## MuJoCo / repo conventions
- MuJoCo 3.12.0: `qM`→`data.M`; `mj_fullM(m,d,dst)` takes `MjData`; `mj_solveM` rows must be size `nv`. Headless render: `MUJOCO_GL=osmesa`.
- Panda from mujoco_menagerie: set `$MENAGERIE_DIR` (menagerie is gitignored). Its `home` keyframe sizes only the 9 arm dofs — `mj_resetDataKeyframe` clobbers an added object's freejoint; set arm joints by `jnt_qposadr` instead.
- `z_local` schema is **frozen** (27-D, v1.1). Mesh-geom curvature is NaN by design (TODO: mesh curvature estimator); primitive grasp contacts are clean.
- Attribution on commits: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Never put a model id in repo artifacts otherwise.

## Where things are
- `mujoco/README.md` — validated-results table + how to run each experiment.
- Pre-embodiment gate: `mujoco/premises_final.py`. Headline experiments: `cross_embodiment_v2.py` (port split + certificate), `surface_field_covering.py` (covering law), `hetero_covering.py` (learned material field), `graspability.py` (graspability certificate), `port_identification.py` (free-space `ε_Y` + two-source certificate).

## Current frontier (update each session)
Certificate now carries both hardware error sources: `ε_Y` (free-space port ID) and `ε_C`
(contact-law sim-to-real). **Next proposed:** a genuine sim-to-real stress test — learn a
law/port under one contact model, evaluate under a deliberately mismatched contact model,
and confirm `ε_C` predicts the degradation. Still-open honest boundary: everything so far is
validated *within-simulator* (errors measured against MuJoCo's own ground truth).
