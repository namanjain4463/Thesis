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
**Correctness pass (2026-09, external audit).** Fixed real bugs, all verified + pushed:
(1) P2 KKT sign — was `(R−W)f = a_u+aref` (residual `−2·a_c`, passed only at a settled state);
corrected to `(W+R)f = aref−a_u` and now tested at a transient too (`premises_final.py`,
`p2_frictionless.py`). (2) `cross_embodiment_v2.py` added a static compliance `1/k` to an
inverse inertia — now `C⁻¹=1/(k·h²)` (consistent 1/kg). (3) covering verdicts now GATE on the
bound-violation fraction. (4) `hetero_covering.py` ground truth (staircase→ramp) fixed → the
learned-field covering **flipped to NEGATIVE**: RBF-KRR mean-reverts under extrapolation
(`μ̂=−0.27`), violates the bound 100%; the covering GEOMETRY still holds. Docs (README §14,
outputs/RESULTS.md) now mark exact-by-construction / synthetic / privileged-info results.
**Open honest boundaries:** learned-field covering needs a Lipschitz-respecting estimator;
`z_local` still ingests true `μ/solref/solimp` (leakage — must become hidden/targets before
training `C_θ`); `Y_G` is the instantaneous (not closed-loop) port; everything within-simulator.
**Next proposed:** sim-to-real stress test (`ε_C` predicts mismatched-model degradation); then
train `C_θ` with a de-leaked dataset + a Lipschitz-respecting field learner.
**De-leaked cross-embodiment transfer (2026-09, audit-corrected).** `deleak_dataset.py` +
`deleak_train_eval.py`: de-leaked `C_θ` (material = categorical id only; raw `μ/solref/solimp`
never fed), trained on the floating gripper, FROZEN, evaluated on HELD-OUT Panda, MATCHED object
distribution, SYNCED logging. A 2nd audit found 3 real issues I fixed: (i) Panda logger combined
post-step qvel with pre-step efc_J → median 20% `v_n` error (fixed: `mj_forward` in `maybe_log`);
(ii) `hoff` sampled but never passed to `run_grasp` (fixed); (iii) frozen vs retrain scored on
different populations (fixed: identical held-out set). **The earlier "port ~doubles transfer
0.24→0.40" DID NOT SURVIVE.** Corrected: the local **compliance** transfers (white-box `F_n=pen/
k_mat`, `R²=0.72`, > Panda-retrain MLP `0.40` — structure beats brute force); the port `W_nn` as a
fitted feature HURTS (`D→C` `0.72→−0.28`; MLPs explode) because float/Panda `W_nn` barely overlap
(extrapolation); absolute `F_n` is grip-confounded (~6× scale). Narrow honest claim: a transferable
local **compliance** law — NOT "the port carries the embodiment", NOT grasp selection. **Open:**
use the real `(W+R)` solve-in-the-loop (not `W_nn` as a feature); move toward grasp DECISIONS
(2nd-reviewer redirection). Backward-compatible `solref_t` added to `scene_xml`/`_wrapper_xml`.
