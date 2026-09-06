# Tier 2 — Certified, capability-aware cross-embodiment grasp selection

> **Status:** research + proposal (no experiments yet). This document synthesizes a literature
> review and proposes the next contribution (Contribution C), positioned to be novel against the
> closest prior work and grounded in what Tiers 0–1 already validated. It follows the 2nd reviewer's
> redirection: *move from contact-force regression to grasp decisions.*

## 0. Provenance & verification honesty

Researched via four parallel literature-review subagents (2026‑09). **arxiv.org and publisher PDF
hosts were egress-blocked**, so every arXiv id below is cross-checked against the search-index result
URLs (often multiple independent hits) but abstract pages were **not** opened. Confidence flags:
`✓` id confirmed via search-index URL; `≈` verified via GitHub/project page or search summary only
(fresh or not-abstract-read); `⚠` from model memory, **not** confirmed — fetch before citing.
Two of the four agents were cut off by a session rate limit; the *cross-embodiment world-model*
cluster is therefore covered more thinly (from the completed briefs + prior knowledge) and is the
first place to deepen the review.

**Correction surfaced by the review:** README §4 cites Neary–Topcu as `arXiv:2412.11215` — that is
**wrong**. The correct paper is **`arXiv:2212.00893`** (L4DC 2023). Fixed in README.

## 1. Landscape (what exists)

### A. Cross-embodiment / gripper-conditioned grasp synthesis
The field conditions grasp generation on gripper **geometry/kinematics**, and generalizes across
grippers by that geometry:
- **GraspGen-X** ≈`2606.00998` (NVlabs, CVPR'26): single diffusion generator conditioned on a
  gripper **swept-volume** descriptor; ~2B grasps, 32 procedural grippers, beats AdaGrasp-TSDF ~25%
  zero-shot on novel grippers. *The single most-overlapping paper with any "gripper-conditioned" claim.*
- **D(R,O) / T(R,O) Grasp** ✓`2410.01702` / ≈`2510.12724`: unified robot–object interaction
  representation; conditions on hand **kinematic description**; kinematically-valid dexterous grasps.
- **MachaGrasp** ✓`2510.06068`: morphology-aware few-shot to unseen hands (eigengrasps/morphology
  embeddings). *Nearest existing "reduced-data on unseen embodiment" claim* — but adaptation signal is
  grasp poses + kinematic morphology, sim dexterous hands.
- **AdaGrasp** ✓`2011.14206`, **UniGrasp** ✓`1910.10900`, **RobotFingerPrint/UGCS** ✓`2409.14519`:
  gripper-aware policy / cross-gripper geometric correspondence.
- Single-gripper detectors (embodiment implicit): **Contact-GraspNet** ✓`2103.14127`,
  **AnyGrasp** ✓`2212.08333` (depth-noise-robust; relevant to D455), **6-DOF GraspNet** ✓`1905.10520`.

### B. Physics/capability/uncertainty-aware grasp selection
- **iTuP / SDG-Net** ✓`2505.01399`: selects grasps by minimizing predicted **interaction wrench**
  along the task trajectory (torque/slip penalties). *Closest "physics not geometry" selection* — but
  single-embodiment mechanics, no transferable contact model.
- **Consensus-driven uncertainty** ✓`2506.20045`: **refuses** grasps when pose-estimator uncertainty
  is high. *Closest "refusal"* — but RGB-pose uncertainty for a fixed gripper, not capability/sensing.
- **GRASPA 1.0** ✓`2002.05017`: graspability score that includes object weight vs robot **payload** —
  a rare explicit capability term, but a post-hoc score, not a selection channel.
- **Reachability-/manipulability-gated ranking** ✓`2103.10562`; multi-criteria ranking **GRaCE**
  ✓`2309.08887`.

### C. Structured / certified contact dynamics
- **ContactNets** ✓`2009.11193` + **Simultaneous learning** ✓`2310.12054` (Posa group): learn a
  physics-consistent contact model of **one object**, robot dynamics assumed known; no embodiment
  split, no transfer bound.
- **Neary–Topcu port-Hamiltonian composition** ✓`2212.00893` (L4DC'23): **Thm 1** bounds composite
  error by subsystem errors — *nearest compositional bound* — but composes subsystems into a bigger
  system (energy ports), not a frozen law onto a new embodiment; no contact, bound not used to decide.
- **Lipschitz-gated planning** ✓`2010.08993` (Knuth et al.): uses model-error Lipschitz + data
  coverage to **refuse to plan** outside a trusted region — *nearest certificate-that-acts* — but
  monolithic model, no contact factorization, no port.
- **Diff-sim contact gradients are often wrong** ✓`2207.05060`; **stiff contact wrecks NN learning**
  ✓`2103.15406` — both **externally corroborate** our decision to use the real `(W+R)` solve rather
  than a learned port feature, and structure over black-box.
- Convex contact solves: **SAP** ✓`2110.10107`, **MuJoCo/Todorov'14** (Delassus source).

### D. Cross-embodiment world models / policies (thinner coverage — deepen later)
- **HPT** ✓`2409.20537` (heterogeneous pretrained transformers), **RoboCraft** ✓`2205.02909`
  (graph-net deformable dynamics); Open‑X/RT‑X, cross-embodiment world models (reviewer's
  `2511.01177`), XIRL/CrossFormer ⚠. These learn **shared latents / policies across embodiments** —
  no physics-factored contact **port**, no certificate, no capability-driven ranking reversal.

## 2. The convergent GAP

Across all four clusters, three things are **absent**:

1. **No one conditions grasp selection on *measured physical capability*.** Every gripper-conditioned
   method conditions on **geometry** (swept volume, URDF, morphology, occupancy, spherical
   correspondence). None feed the arm's reach envelope + joint-torque/payload feasibility + the
   **contact-space inverse-inertia (Delassus port)** — the physical "how hard can THIS body push at
   THIS contact through its real mechanics/controller" — into *which grasp/hand is chosen*.
2. **No transferable, embodiment-factored contact model drives selection.** Physics-based selectors
   (iTuP, force-closure) use a **single-embodiment** mechanics model retrained per robot. The
   structured-contact line (ContactNets, Neary–Topcu, Knuth) never does the `W = Y_object + Y_robot`
   split, never freezes a local law and recomposes with a per-embodiment analytical port, and never
   turns a transfer bound into a **grasp go/no-go**.
3. **No capability-and-sensing refusal.** Only pose-uncertainty refusal exists. No method refuses a
   grasp because *this embodiment cannot deliver the required contact wrench*, or because
   *camera-only sensing cannot verify the internal squeeze the task needs* (our internal-force
   nullspace result).

## 3. Proposed Contribution C — **Port-Conditioned Certified Grasp Selection (PC‑CGS)**

**One sentence.** *A single frozen interaction model that selects grasp location and hand
participation by composing an embodiment-invariant learned contact law with each body's **recomputed
analytical contact port** (Delassus) inside the real convex solve, and gates the choice with a
two-source transfer certificate plus a sensing-observability test — so the grasp ranking **reverses
correctly and for a certified physical reason** when the embodiment changes, refusing grasps that are
infeasible or unobservable for that body, with **near-zero new-body grasp data.***

### 3.1 The load-bearing insight (and why our own Tier‑1 negative *motivates* it)
The entire grasp-synthesis literature is missing a **capability** channel; it uses gripper geometry.
Our validated result supplies exactly that channel physically: **the analytical port `Y_robot`
(Delassus `W = Y_object + Y_robot`) is the missing measured-capability signal** — it encodes the
body's force/inertia/controller response at a contact, is **computable from the robot model with no
grasp data**, and is what *must be recomputed per embodiment* (Tier‑0/1). Tier‑1 also told us **how
to use it**: as a fitted MLP feature the port *hurt* (non-overlapping distributions → extrapolation),
while the frozen **local compliance transferred** (white-box `R²=0.72`). Conclusion — **put the port
where physics puts it: inside the `(W+R)` solve, not as a learned input.** PC‑CGS does exactly that,
and `2207.05060` (bad diff-sim contact gradients) independently backs this choice.

### 3.2 The model (three legs)
For a candidate grasp `a = {contact patches, hand assignment, approach, closing, transport}` on
object estimate `b_t` with embodiment `E`:

1. **Port-conditioned interaction rollout.** `a` + object geometry → contact set → the analytical
   port `Y_G(a,E)` (per-embodiment, recomputed, no learning). Predicted contact wrench / object motion
   `= Solve(frozen C_θ(z_local), Y_G(a,E))` via the convex `(W+R)` solve. Rank grasps by a task cost
   `J` through this composed interaction — **mechanics, transferred, not geometry**.
2. **Certificate gate (accept / probe / reject), each with a physical reason.** Per candidate compute
   (i) **kinematic** margin `γ_kin` (reach/finger-span), (ii) **wrench/force-limit** margin — can
   `Y_G` + actuator limits deliver the task wrench (friction-cone `γ_fric`), (iii) the **transfer
   certificate** `‖ΔH‖ ≤ (ε_Y+ε_C)/(m(m−ε))`, `m=σ_min(Y_G+C⁻¹)` — does the frozen law transfer to
   this port within tolerance, (iv) the **observation/internal-force test** `ker O ⊆ ker P_τ` — can
   this body's sensing (camera-only ⇒ no squeeze) verify what the task needs; if not and the task is
   force-critical → **reject**. Output a calibrated decision naming the binding margin.
3. **Certified ranking reversal.** Because `Y_G` is recomputed analytically per body, the **same
   frozen model** yields a **different best grasp** when the embodiment changes — and the margins
   *name why* (reach vs force-limit vs balance vs unobservable). The change is **certified**, not
   merely learned.

### 3.3 Why this is novel (guardrails satisfied, per closest work)
- vs **GraspGen-X / D(R,O)/T(R,O)** — they condition on gripper **geometry/kinematics**; PC‑CGS
  conditions on the analytical **contact port** (a *physical capability*) and ranks by a *transferable
  contact-mechanics* outcome, not a learned geometric score.
- vs **MachaGrasp** — data-efficiency comes from the **analytical port swap** (zero new grasp data for
  the new body's capability), reported as a measured adaptation-data delta on a *physically distinct
  real arm* (Lynxmotion, no force sensing), not sim-family morphology fine-tuning.
- vs **Consensus-uncertainty / AnyGrasp** — refusal is **capability-and-sensing** driven (can't
  deliver the wrench / can't observe the needed internal force), from the internal-force nullspace +
  port, not pose-uncertainty robustness.
- vs **iTuP / force-closure** — uses a **transferable, embodiment-factored** contact model (frozen
  `C_θ` + per-embodiment port) with a **certificate**, not a single-embodiment mechanics net.
- vs **Neary–Topcu** — a **two-source (ε_Y+ε_C)** bound on the **Delassus contact operator** used for
  a **grasp decision**, not an energy-composition subsystem bound for prediction.
- vs **Knuth (Lipschitz-gated)** — the gate drives **grasp selection/refusal** on a **factorized**
  contact model, not a monolithic-model trust region for planning.
- vs **ContactNets** — splits `W=Y_obj+Y_robot` and **freezes the local law across embodiments**;
  ContactNets keeps robot dynamics fixed and models one system.

## 4. The decisive experiment — **certified grasp-ranking reversal**

Same object + task; change the embodiment; the correct grasp must change for a physically
identifiable reason, and the frozen model + certificate must predict the change **and its reason**.

**Bodies (increasing distance):** floating 2‑finger → Franka Panda → a Lynxmotion-like low-payload
3‑finger arm (camera-only). **Later:** a bimanual rig and a humanoid (shared base) — the port simply
becomes the multiport `Y_G = Bᵀ blkdiag(Y_1..Y_n,Y_obj) B`.

**Reversal scenarios (each isolates one physical cause):**
| Scenario | Correct behavior | Binding margin |
|---|---|---|
| Off-center CoM, heavy | shift grasp / share load; weak body must pick a deliverable grasp or refuse | wrench/force-limit |
| Object reachable by one body only | different approach/grasp | kinematic `γ_kin` |
| Slippery + heavy (force-critical) | force-sensored body accepts; **camera-only body REJECTS** | observation `ker O ⊆ ker P_τ` |
| Two inadequate single grasps | recruit 2nd hand — but do **not** assume a 2nd hand guarantees success | hand-participation + wrench |
| Denser contact sampling | **no** artificial increase in predicted holding capacity | certificate stability |

**Metrics (decisions, not force R²):** selected-grasp success & task completion in full sim;
**ranking-reversal accuracy** (does the frozen model's argmax match ground-truth feasibility across
bodies) and **reason accuracy** (does the binding margin match the true cause); **false-accept /
false-reject**; **unnecessary second-hand use**; and the **adaptation-data curve** — new-body data to
match a per-body-retrained selector (PC‑CGS target ≈ 0 for the capability channel).

**Baselines:** (i) geometry-conditioned selector (GraspGen-X-style proxy), (ii) per-body **retrained**
selector (reference), (iii) analytical-only (no learned law), (iv) single-embodiment physics selector
(iTuP-style). This separates *value of the factorization* from *value of learning* and from *value of
supplied physics* — the same three-way separation the 2nd reviewer demanded in Tier‑1.

## 5. Honest risks & limits
- **Grip-strategy / operating-point confound** (seen in Tier‑1, ~6× force scale) must be controlled by
  ranking on **grip-invariant** quantities (task feasibility, compliance) not absolute force.
- **`Y_G` is still the instantaneous/rigid port**; the closed-loop controller-shaped port (esp.
  Lynxmotion servo) is unidentified — the certificate's `ε_Y` must carry this, and reject when large.
- **Everything within-simulator** until a sim-to-real `ε_C` stress test exists (Tier‑0 open item).
- **Reduced-data claim is the risky one** — it must be a *measured* delta on a physically distinct
  body, not a sim-family number (the MachaGrasp trap).
- **No standardized benchmark** for capability-dependent ranking reversal across embodiments was found
  — we would define one (a contribution, but also a burden of proof: baselines must be strong).

## 6. First concrete build — **DONE** (`mujoco/grasp_ranking_reversal.py`)
A MuJoCo **grasp-ranking-reversal micro-benchmark** on the force/moment-capability axis. Off-center-CoM
bar; candidate grasps along it; two embodiments = a strong vs a weak gripper (grip force is a real
hardware capability). **Result:** the MuJoCo-ground-truth best grasp **reverses** — strong body →
geometric-center grasp (y=0), weak body → a grasp toward the CoM (y=0.02) because y=0 tips out of its
weaker grip. **PC‑CGS** (one geometric finger-lever fit as a single constant on the pooled ground
truth + each body's rated grip force) predicts **both argmaxes correctly (2/2)**, reproduces the
feasibility map **9/10**, and **names the binding reason** — the *moment* margin (−0.039 N·m for the
weak body at y=0). A **geometry-only** ranker picks y=0 for both and is **wrong for the weak body**.
Figure + log: `outputs/rankrev_reversal.{png,txt}`.

**Honest scope of v1:** the two bodies share the gripper morphology and differ in *rated grip force*
(same port/reach) — this isolates the force/moment axis cleanly; deliverable force is the rated spec
(`2×squeeze`), not the measured Delassus port; the moment model is a single lumped lever, not the full
`(W+R)` solve. **Next (v2):** the *reach/manipulability* axis with the real **Panda** (a genuinely
different port), the *observability/refusal* axis (camera-only → reject force-critical), and the
*bimanual* hand-participation decision — each a distinct physical cause of reversal.

## 7. References (confidence-flagged)
Grasp synthesis: GraspGen-X ≈2606.00998 · D(R,O) ✓2410.01702 · T(R,O) ≈2510.12724 · MachaGrasp
✓2510.06068 · AdaGrasp ✓2011.14206 · UniGrasp ✓1910.10900 · RobotFingerPrint ✓2409.14519 ·
Contact-GraspNet ✓2103.14127 · AnyGrasp ✓2212.08333 · 6-DOF GraspNet ✓1905.10520 · GenDexGrasp
✓2210.00722 · DexGraspNet ✓2210.02697 · GraspGen ✓2507.13097.
Selection/uncertainty: iTuP/SDG-Net ✓2505.01399 · Consensus-uncertainty ✓2506.20045 · GRASPA
✓2002.05017 · reachability-ranking ✓2103.10562 · GRaCE ✓2309.08887.
Contact/certified: ContactNets ✓2009.11193 · Simultaneous-learning ✓2310.12054 · **Neary–Topcu
✓2212.00893** · Knuth Lipschitz-gate ✓2010.08993 · diff-sim-gradients ✓2207.05060 · stiff-contact
✓2103.15406 · SAP ✓2110.10107 · Nimble ✓2103.16021 · GNS ✓2002.09405 · TossingBot ✓1903.11239.
World models: HPT ✓2409.20537 · RoboCraft ✓2205.02909 · cross-embodiment WM (reviewer) 2511.01177 ·
XIRL/CrossFormer ⚠. From-memory (fetch before citing): HNN ⚠1906.01563 · DeLaN ⚠1907.04490 · Kloss ⚠1710.04102.
