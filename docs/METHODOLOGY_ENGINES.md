# The Three Topology / Electrical Engines of the Integrated Planning Workflow

**Purpose of this document.** A precise, implementation-grounded description of the three
methodological engines that the workflow chains together:

1. **LV radial topology** — the *pole-first* heuristic that turns a set of georeferenced
   customers + roads into poles, service drops and a low-voltage backbone.
2. **MV topology & transformers** — how the settlement is split into transformer service
   zones, how transformers are **placed, sized and assigned**, and how the medium-voltage
   trunk is routed and validated.
3. **SHS economic screening** — how *terminal subtrees* (leaf branches) of the grid are
   enumerated and economically screened to carve out solar-home-system (hybrid) pockets.

This is deliberately **engine-by-engine**, not workflow-level: how they are orchestrated
(A→C→LV-screen→B→MV/LV-screen→D) is covered elsewhere. Each engine section gives (a) a short
discursive paragraph usable verbatim in the talk, (b) the exact method and equations as
implemented, (c) the electrical assumptions baked in, and (d) the **open questions** most worth
putting to a power-systems supervisor.

Units: power **kW/kVAr/kVA**, energy **kWh**, length **m**, voltage **kV** (per-unit `pu`
where noted), money in generic monetary units (MU). All source citations point to the current
code.

---

## 0. Shared electrical foundation (context for all three engines)

All three engines are validated on the **same** steady-state model, so it is worth stating once.

- **Balanced three-phase steady-state power flow** (pandapower `runpp`, flat start, shunt
  capacitance neglected `c_nf_per_km = 0`). No time series — every check is a **peak snapshot**.
- **Per-customer load** at a snapshot (`pandapower_builder._pq_kw`):

  $$P_i = P_i^{\text{now}}\,(1+g_i)^{T}\cdot cf, \qquad Q_i = P_i\tan\!\big(\arccos pf_i\big),\quad pf_i\in[0.5,1.0]$$

  with growth $g_i$, horizon $T$, and a **coincidence factor** $cf$. Loads are aggregated to the
  pole the customer is tied to: $P_{\text{pole}}=\sum_{i\in\text{pole}}P_i$.
- **Three snapshots** are evaluated per candidate: *now* (design, $cf=0.6$, no growth), *horizon*
  (design, $cf=0.6$, with growth — **this is what gates feasibility**), and a *coincident-peak
  stress* ($cf=1.0$, with growth — **reported only, never gates**).
- **Feasibility** = power flow converged **and** $v_{\min}\ge 0.90$, $v_{\max}\le 1.05$,
  line loading $\le 100\%$, transformer loading $\le 100\%$ (`electrical/validation.py`).
- **Iterative conductor sizing** (`conductor_sizing.size_conductors`): start every line on the
  cheapest conductor; each iteration, upgrade the **critical lines** (all overloaded lines **plus**
  every line on the shortest path from the slack to any undervoltage bus) by one catalogue rank;
  stop when feasible, when no larger conductor exists, at 8 iterations, or after 2 consecutive
  non-convergences (voltage collapse cannot be reconductored away). This same loop sizes the flat
  LV, each MV/LV zone, and (by uniform rank sweep) the MV trunk.

> **Speech line:** *"Everything downstream rests on a balanced three-phase snapshot model.
> We never optimise topology and electrics simultaneously — we generate a geometry heuristically,
> then let a deterministic reconductoring loop repair it until the power flow passes. That
> separation is the single most important modelling choice, and it is where most of my open
> questions live."*

---

## 1. Engine 1 — LV radial topology (pole-first heuristic)

*Source: `core/distribution_service.py::_run_low_voltage_core`, `core/distribution_algos.py`,
`core/topology/lv_topology.py`.*

### 1.1 Discursive summary (for the talk)

> *"The LV engine is 'pole-first' and purely geometric. Roads are sampled at a fixed spacing to
> propose candidate poles; each customer is buffered and greedily associated to the nearest
> feasible pole under a service-drop radius and a max-customers-per-pole cap; buildings that no
> road pole can reach get their own free-placement poles at cluster centroids. The retained poles
> are then joined into a single **radial backbone by a Minimum Spanning Tree** — minimising total
> conductor length (optionally penalised for going off-road). Only after the tree exists do we run
> the power flow and size conductors. So the backbone minimises **length**, not voltage drop or
> losses — those are handled reactively by reconductoring, never by moving a wire."*

### 1.2 Method, step by step

**(1) Candidate poles from roads** (`collect_sampled_points`). Every road line is sampled at
`sampling_distance = road_pole_spacing_m` to yield candidate pole points.

**(2) Customer→pole association** (`associate_buildings_to_poles`). Iterating pole by pole, a
buffer of radius `user_distance = max_user_connection_radius_m` selects still-unassociated
customers; they are ranked and up to `max_associations = max_users_per_pole` are attached:

$$\text{rank key} = \big(\underbrace{-\,\text{priority score}}_{\text{high first}},\;
\underbrace{d(\text{cust},\text{pole})}_{\text{near first}},\; \text{building index}\big)$$

With uniform priority this reduces to nearest-first. Poles that end up serving nobody are dropped.

**(3) Free poles for the remainder** (`place_poles_for_unassociated_buildings`). Customers no road
pole could reach are clustered greedily: repeatedly take the **highest-priority / largest** cluster
(buildings within a mutual `user_distance` buffer), place a pole at the **centroid of the merged
buffers**, and attach the feasible members (those within the radius after the pole is fixed),
capped at `max_associations`. Fallbacks anchor the pole directly on a building when the centroid
violates the radius; must-connect customers are protected from being stranded.

**(4) Backbone by MST** (`create_graph_and_mst`). A **complete graph** over all retained poles is
built and reduced to its Minimum Spanning Tree. Each candidate edge $(a,b)$ carries a *physical*
length and a *selection weight*:

$$\ell_{ab}=\lVert a-b\rVert \quad(\text{projected, metres}), \qquad
w_{ab}=\ell_{ab}\Big(1+\lambda\, f^{\text{off}}_{ab}\Big)$$

where in **euclidean** mode (the MVP default) $w_{ab}=\ell_{ab}$, and in **road-attracted** mode
$f^{\text{off}}_{ab}$ is the fraction of sampled points on the edge lying farther than a buffer from
any road and $\lambda$ is a penalty factor. The MST minimises $\sum w$, i.e. total (penalised)
length. This is a tree ⇒ **strictly radial**, no loops.

**(5) Span densification** (`densify_mst_edges`). Edges longer than `max_pole_span_m` are split by
inserting intermediate poles (routed along roads when available). These poles are **mechanical
only** — they carry no load, they just bound the span.

**(6) Packaging** (`lv_topology._package_topology`). The result becomes an `LVTopology`: a rooted
`networkx` graph, `pole_xy`, `customer_pole`, `pole_customers`, and per-customer **service-drop
length** $s_i = d(\text{building}_i,\text{pole}(i))$. The **slack pole** is the pole nearest the
source point. Total LV length $= \sum_{e}\ell_e$.

### 1.3 Electrical assumptions embedded here

| Assumption | Consequence |
|---|---|
| Backbone = **min-length MST** | Optimises copper, **not** $\Delta V$ or $I^2R$ losses. A min-length tree ≠ min-loss tree. |
| **Radial** only | No meshing, no loop flows, no N-1 reliability — standard for rural LV but unmodelled as a *choice*. |
| Association & pole placement are **greedy** | Deterministic, fast, but not a global facility-location optimum. |
| **Service drops carry no impedance** | The drop is priced (via cheapest LV conductor) but its voltage drop is **ignored** in the power flow — loads sit directly on the pole bus. |
| **Balanced 3-phase**, 3-phase ABC conductors | No phase allocation / no single-phase laterals, no neutral / no unbalance — often the dominant reality in rural LV. |
| **Ideal slack** at the source pole, $v=1.0\,$pu | No source/upstream impedance for the flat-LV cases. |
| Loads aggregated **to the pole** | Intra-pole diversity handled only through the global $cf$. |

### 1.4 Open questions for the supervisor (Engine 1)

1. **Geometry–electrics decoupling.** The topology is chosen by length and only *repaired*
   electrically by reconductoring. Is a length-MST + reconductoring loop an acceptable proxy for a
   voltage-/loss-aware backbone, or should the edge weight embed an electrical term (e.g. weight by
   $\ell\cdot(1+\alpha\,I^2 R)$ or a downstream-kVA·length moment)? How much do we lose vs. a
   loss-minimising / voltage-constrained tree?
2. **Service-drop impedance.** We neglect the voltage drop on the final drop to each customer. In
   rural LV the drop can be a non-trivial share of the total $\Delta V$. Worth adding as a lumped
   impedance per customer, or is pole-lumping fine at planning altitude?
3. **Unbalance & single-phase laterals.** The whole model is balanced 3-phase. For SSA rural LV,
   single-/split-phase spurs dominate. Is a balanced approximation defensible for a *planning*
   comparison, or does it systematically flatter the LV option?
4. **Coincidence factor.** A single flat $cf=0.6$ for design (and 1.0 for stress). Should this be
   load-class dependent and size-dependent (ADMD / Velander $P=a\,n+b\sqrt n$), especially where a
   few institutional loads dominate?
5. **Reliability.** Radial trees give no redundancy. Do we need any reliability metric (SAIDI-like,
   or at least longest single-point-of-failure feeder length) as a comparison axis?

---

## 2. Engine 2 — MV topology & transformers

*Source: `core/topology/mvlv_service_zones.py`, `core/electrical/pandapower_builder.py::build_mvlv_net`,
`core/workflow/mvp_runner.py::_build_mvlv`.*

### 2.1 Discursive summary (for the talk)

> *"When a single 0.4 kV network can't carry the load without collapsing, we partition the **whole**
> settlement into transformer zones. The **number of transformers `k` is chosen by a selectable
> criterion**, not left as a byproduct of packing: `voltage_cap` searches `k` over a **nested** family
> of partitions and lets the power flow pick the smallest count that holds the LV voltage drop **with a
> declared margin**, falling back to the smallest merely-feasible count when no count can reach it;
> `distance_cap` cuts a complete-linkage dendrogram so every cluster diameter is **≤ the cap by
> construction**; `capacity` is the legacy greedy farthest-first packing. **All three respect the
> transformer kVA cap** (any cluster exceeding the largest procurable rating is split). Each zone gets
> one transformer at its **centroid, snapped to the nearest road**, sized to the zone's **coincident**
> horizon kVA. Transformers are tied to the source by an **MST medium-voltage trunk**. The plant is now
> modelled honestly: a **single slack on the plant LV bus** at the sending voltage, an **explicit
> step-up transformer** (LV→MV, sized to the settlement's system peak, allowed to exceed the catalogue)
> feeding the MV backbone, and step-downs at each zone — replacing the earlier shortcut of a fixed
> 1.04 pu slack placed directly on the MV bus."*

### 2.2 Method — zoning (criterion-driven `k`)

The number of transformers `k` (= number of zones) is set by `clustering_criterion`
(`core/topology/mvlv_service_zones.py::partition_zones`, k-search in
`core/workflow/mvp_runner.py::_build_mvlv`). **The kVA cap below is enforced under every criterion**
via `enforce_kva_cap`, which recursively bisects any zone whose coincident apparent power exceeds it.

**Capacity cap.** From the transformer catalogue, the largest derated rating divided by the sizing
safety factor:

$$S_{\max}^{\text{zone}} = \frac{\max_t\big(sn^t_{kVA}\cdot \text{max\_loading}^t/100\big)}{1.2}$$

**(a) `voltage_cap` (default) — power-flow-driven search over a nested family.** Seed `k` at
`suggest_zone_start` (complete-linkage cluster count at a "hopeless" 2 km diameter). Each attempt cuts
**one Ward dendrogram** at `k` (`partition_hierarchical_k`), enforces the kVA cap, builds the full
MV/LV network and solves it.

The partition family is **nested**: the `k+1` partition *refines* the `k` partition, so a zone is
split rather than reshuffled. This replaced k-means-per-`k`, which re-solved from scratch at every
`k` and made the worst voltage drop **non-monotone** in `k` — measured on Mafa, `k=4` gave a 0.1028 pu
drop against `k=3`'s 0.1004 pu. A search that stops at the first failure cannot rest on that.

A count is **accepted** at $v_{\min}\ge$ `voltage_min_pu` $+$ `design_voltage_margin_pu` (default
0.02 pu), not merely at the feasibility limit: minimising `k` against the bare limit spends every volt
of headroom and leaves the accepted design sitting on the constraint (0.913 pu against a 0.90 limit,
measured).

The margin is a **preference, not a constraint**. More transformers shorten LV feeders but lengthen
the MV trunk and insert another transformer impedance, so the achievable voltage **plateaus** — on
Mafa no zone count reaches a 0.92 pu target. When the margin is unreachable the search returns the
**smallest merely-feasible** count and records why; with nothing feasible it reports the
least-violating partition. `k` never exceeds the customer count (the old climb was bounded only by
`max_transformers` and crashed out of the partitioner on any smaller settlement); diverged attempts
step geometrically; and the climb stops once extra transformers stop buying voltage — but only once
something feasible is in hand, because on Mafa the voltage sits flat at 0.884 pu for `k=4..7` and then
*jumps* to 0.905 at `k=10`, so flat is not saturated. `k_selection_snapshot="coincident"` selects
against the coincident peak instead.

The search is the dominant runtime cost: one full plan build, per-zone sizing loop and MV rank sweep
per attempted `k`, for Case C and again for Case D (~240 s for 240 customers without numba).

**(b) `distance_cap` — geometric guarantee, one-shot.** Complete-linkage hierarchical clustering cut at
`max_cluster_diameter_m`, so every cluster diameter is **≤ the cap by construction**; `k` emerges from
the cut, no search and no power flow (deterministic). Errors if the cap needs `k >` `max_transformers`.

**(c) `capacity` — legacy greedy farthest-first** (`partition_service_zones`, deterministic):

1. Seed = unassigned customer **farthest from the source** (ties broken by id).
2. Grow by nearest customer $c$ to the seed while **all** hold:
   $d(c,\text{seed}) \le$ `max_lv_zone_radius_m` (250 m default), members $<$ `max_zone_customers`
   (60 default), and $\sum |S| + |S_c| \le S_{\max}^{\text{zone}}$.
3. Repeat until every grid customer is assigned. Here `k` is a *byproduct* of packing; the voltage
   check is informative only.

Because the neighbour list is distance-sorted, the radius test short-circuits the growth.

### 2.3 Method — transformer placement, sizing, assignment

**Placement** (`_transformer_point`). The transformer sits at the **unweighted centroid** of the
zone's customer points, then **snapped to the nearest road** geometry (fallback: the raw centroid).
The transformer LV winding is tied to the zone's **slack pole** (the pole nearest the transformer)
by a closed zero-impedance bus–bus switch.

**Sizing** (`zone_apparent_power_kva` + `select_transformer`). Zone apparent power uses the
**coincident** horizon peak (vector sum of P and Q, **$cf=1.0$**, i.e. no diversity — conservative
on purpose):

$$P_z=\sum_{i\in z}P_i^{\text{now}}(1+g_i)^{T},\quad
Q_z=\sum_{i\in z}P_i\tan(\arccos pf_i),\quad
S_z=\sqrt{P_z^2+Q_z^2}$$

Then pick the **smallest-rank** transformer that covers a safety-margined target, else the largest:

$$\text{target}=1.2\,S_z,\qquad
\text{choose min-rank }t:\; sn^t_{kVA}\cdot\tfrac{\text{max\_loading}^t}{100}\ \ge\ \text{target}.$$

The catalogue is discrete (e.g. Mafa: 100 / 200 / 315 / 500 kVA). **Assignment** is one transformer
per zone; each customer inherits its zone's transformer id.

### 2.4 Method — MV trunk & network validation

**MV routing** (`_build_mv_topology`). Pseudo-nodes = {source} ∪ {one node per transformer}; the LV
topology engine (Engine 1, road-attracted if roads exist) is reused to route an **MST trunk** over
them. Each transformer's MV feed length = source→pole path length + its MV service drop.

**pandapower model** (`build_mvlv_net`). MV buses at `mv_voltage_kv` (11 kV), LV buses at 0.4 kV.
A **single** `ext_grid` slack sits on the **plant LV bus** at `source_vm_pu` (default **1.0**, the honest
sending voltage). An **explicit step-up transformer** (`stepup_tx`, LV→MV) energises the MV backbone; a
single-slack invariant is asserted after build. Each zone: MV line (single chosen conductor) →
step-down `create_transformer_from_parameters` ($sn_{MVA}=sn_{kVA}/1000$, catalogue
$v_k\%, v_{kr}\%, p_{fe}, i_0\%$) → the zone's local LV tree. The step-up is sized to the settlement's
coincident system peak (`build_stepup_spec`), may exceed the catalogue (a project-procured unit), and its
CAPEX is included in the transformer cost.

**MV conductor sizing** (`_build_mvlv`). The zone LV conductors are sized first (Engine-0 loop, per
zone). Then the **MV conductor is swept by rank** — the whole trunk uses **one** conductor type —
rebuilding and re-solving the full network until it is feasible.

### 2.5 Electrical assumptions embedded here

| Assumption | Consequence |
|---|---|
| **Explicit step-up transformer** (updated) | Single slack on the **plant LV bus** at `source_vm_pu` (default 1.0); an explicit LV→MV step-up feeds the MV backbone with real impedance. Retires the fixed 1.04-pu-on-MV-bus shortcut (the OLTC is still not modelled). |
| **`k` chosen by criterion** (updated) | `voltage_cap` searches a **nested** Ward family for the smallest `k` holding the voltage limit **plus a declared margin**, falling back to the smallest feasible `k` when the margin is unreachable; `distance_cap` guarantees cluster diameter ≤ cap; `capacity` is the legacy packing. The kVA cap is enforced under all three. Zones still ignore *joint* MV-length minimisation. |
| Transformer at **unweighted centroid** | Not the kVA-weighted load centroid, so intra-zone LV $\Delta V$ isn't minimised by placement. |
| Sizing on **coincident $cf=1.0$**, network validated at **$cf=0.6$** | Now an explicit parameter (`transformer_sizing_coincidence_factor`), not a hard-coded constant. The rating is $1/cf$ larger than the snapshot it is checked against, so its loading peaks near 50% and the loading limit cannot bind. Tracking the design factor instead is **not** simply better: `coincidence_factor` is flat and system-wide, diversity only emerges with many customers, so it under-sizes small zones — and a smaller unit has a *higher* ohmic impedance ($v_k\%/sn$), costing ~0.010 pu of LV voltage (measured: it flipped Case D from feasible to infeasible). A size-dependent ADMD curve is the real fix. |
| **One 1.2 margin, two uses** (corrected) | The 1.2 in `S_max^zone` and the 1.2 in `select_transformer` are the **same** margin, not a compounded 1.44: the cap exists so a zone sitting at the cap stays coverable by the largest catalogue unit. Named `TRANSFORMER_SIZING_SAFETY_FACTOR`. |
| **One MV conductor for the whole trunk** | No per-segment MV sizing; coarse but simple. |
| Bus–bus **switch** ties trafo LV to slack pole | Zero-impedance tie; no LV busbar / no in-zone MV reticulation. |
| Single **radial** MV feeder, one source | No ring / no backup / no second injection. |
| Reactive power | No compensation / no capacitor banks; Q is drawn from the slack. |

### 2.6 Open questions for the supervisor (Engine 2)

> **Update.** Q1 is **addressed** (explicit step-up, single LV-bus slack). Q2 is **partly**
> addressed: `k` now comes from a power-flow search over a *nested* Ward family with a declared
> voltage margin and a documented fallback, or from a `distance_cap` diameter guarantee, both
> kVA-capped — but zones are still not formed by *jointly* minimising MV trunk length. Q4 is
> **reframed, not closed**: the sizing coincidence basis is now an explicit parameter and the "double
> margin" turned out to be one margin used consistently, but the flat-vs-zone-scale diversity problem
> underneath it is real and needs a size-dependent ADMD curve. OLTC / $X_{source}$ and loss-driven
> sizing remain open.

1. **~~The 1.04 pu source & missing step-up transformer.~~ (addressed)** The step-up is now explicit and
   the slack sits on the plant LV bus at the sending voltage. Remaining: what sending voltage /
   `X_source` / OLTC behaviour to assume for the upstream primary substation?
2. **Geometric vs. electrical zoning. (partly addressed)** `voltage_cap` now sets `k` from the actual
   power flow over a nested partition family, and `distance_cap` from a hard diameter bound; zones are
   still not formed by *jointly* minimising MV trunk length. Is a Ward cut + margin search sufficient,
   or move to a proper capacitated facility-location / K-median with an explicit MV-length objective?
   Note the measured plateau: past ~10 zones on Mafa, extra transformers buy no voltage because the
   binding drop has moved to the MV trunk and the step-up — which suggests the MV side, not the
   zoning, is where the remaining headroom is.
3. **Transformer siting.** Centroid-snapped-to-road vs. **kVA-weighted load centroid** (which
   minimises $\sum kVA\cdot\ell$ inside the zone). Is the LV voltage penalty of the unweighted
   centroid material?
4. **Coincidence basis for transformer sizing. (reframed)** The cap and the sizing target share one
   1.2 margin, so there is no stacked 1.44. What remains is the real question: `coincidence_factor` is
   a single flat system-wide number, so neither $cf=1.0$ (over-sizes, loading limit never binds) nor
   $cf=cf_{design}$ (under-sizes small zones, costs LV voltage) is right at zone scale. What
   size-dependent ADMD / diversity curve should set the sizing peak as a function of the number of
   aggregated customers?
5. **Discrete catalogue & losses in sizing.** Transformer choice ignores its own no-load ($p_{fe}$)
   and load losses in the *economic* sizing — only the CAPEX ($\text{cost\_fixed}+\text{cost\_per\_kVA}\cdot sn$)
   enters. Should lifetime loss cost drive the sizing/selection, not just the loading constraint?
6. **Uniform MV conductor & single feeder.** One conductor for the entire trunk and a single radial
   feed. For longer/loaded trunks, would per-segment sizing or a ring materially change the answer?

---

## 3. Engine 3 — SHS economic screening (terminal subtrees)

*Source: `core/topology/subtree_candidates.py`, `core/topology/branch_detection.py`,
`core/economics/{standalone,npc,cost_model}.py`, `mvp_runner._avoided_grid_cost` /
`_select_standalone_clusters`.*

### 3.1 Discursive summary (for the talk)

> *"Instead of asking 'which customers look remote', we enumerate the **terminal subtrees** of the
> radial grid — every maximal leaf branch that detaches at a fork node — and ask a purely economic
> question of each: is the net present cost of solar home systems for **all** its customers cheaper
> than the **marginal grid cost we'd avoid** by cutting that branch off? Because subtrees nest — a
> remote hamlet contains its own stubs — we screen coarsest-first and let a claimed-set stop
> double-counting. Crucially the decision is threshold-free: the old 'too far / too long' rules are
> now diagnostics only. The screen runs identically on the flat-LV tree (giving Case B) and inside
> each transformer zone's LV tree (giving Case D)."*

### 3.2 Method — candidate enumeration

**Terminal subtrees** (`terminal_subtrees`). Root the pole graph at the slack. A **fork** is any
node with $\ge 2$ children. For each fork $B$ and child $c$, the candidate is the set of poles
**at/below $c$**, with entry edge $(B,c)$. Properties, by construction:

- Trunks / pass-through chains (one child) never spawn a candidate ⇒ the un-cuttable backbone, and
  the whole zone, are **excluded**.
- Subtrees **nest** and form a **laminar family** (any two are nested or disjoint).
- A pure chain with no fork yields **no** candidate.

**Per-subtree metrics** (`subtree_clusters`). Branch edges = internal edges ∪ entry edge. Then:

$$\ell^{\text{br}}=\!\!\sum_{e\in\text{branch}}\!\!\ell_e,\quad
C^{\text{br}}=\!\!\sum_{e\in\text{branch}}\!\!\ell_e\cdot \text{cost/m}_e\cdot m_{\text{civil}},\quad
s^{\text{br}}=\!\!\sum_{i\in\text{sub}}\!\! s_i,\quad
P^{\text{br}}_{\text{hor}}=\!\!\sum_{i}\! P_i^{\text{now}}(1+g_i)^T$$

with civil multiplier $m_{\text{civil}}=1.2$. For **Case B** the branch also slices min-voltage,
max-loading and branch $I^2R$ losses from the solved full-LV power flow; for **Case D** zones these
are not sliced (no per-zone PF cut). The old distance/path thresholds survive only as **diagnostic
flags** (`_score`) and never gate anything.

### 3.3 Method — SHS eligibility & package NPC

**Eligibility** (subtree passes only if **every** customer passes — one veto blocks the branch):

$$\text{must\_connect}_i=\text{False},\quad \text{can\_be\_standalone}_i=\text{True},\quad
\exists\,\text{pkg}:\ \text{tier}\ge d_i,\ \max\!P_{kW}\ge P_i^{\text{now}}(1+g_i)^T,\ \text{annual\_kWh}\ge \text{kWh}_i$$

**Cheapest eligible package per customer**; cluster NPC $=\sum_i$ package NPC. Package NPC
(`standalone_npc`) over horizon $T$ at rate $r$:

$$\text{NPC}_{\text{pkg}} = \text{capex} + f_{OM}\,\text{capex}\cdot AF(r,T)
+ \!\!\sum_{t=L,2L,\dots<T}\!\! \frac{f_{\text{repl}}\,\text{capex}}{(1+r)^t}
-\frac{\text{salvage}}{(1+r)^T},\quad AF(r,T)=\frac{1-(1+r)^{-T}}{r}$$

with lifetime $L$, replacements at each full lifetime strictly before $T$, and a linear
residual-life salvage credit.

### 3.4 Method — avoided grid cost & selection

**Marginal avoided grid cost** if a subtree goes off-grid (`_avoided_grid_cost`):

$$\text{avoided}= C^{\text{br}}
+ s^{\text{br}}\cdot\text{cost/m}^{\text{LV}}_{\text{cheapest}}\cdot m_{\text{civil}}
+ n\big(c_{\text{conn}}+c_{\text{meter}}\big)
+ n_{\text{poles}}\,c_{\text{pole}}\,m_{\text{civil}}
+ \underbrace{\text{PV}\big(\text{gen LCOE}\cdot\!\sum_i \text{kWh}_i\big)}_{\text{avoided grid generation, BOTH screens}}
\;\big[\,+\,\text{PV}(\text{branch losses})\,\big]$$

- **Avoided grid generation** (updated) is now credited in **both** screens: once a subtree goes
  off-grid the mini-grid no longer generates its energy, so the PV of that generation (same LCOE and
  basis as the grid architectures' `generation_npc`) is a genuine saving. It enters *only* the
  selection gate, so no total is double-counted; it makes the gate the true total-NPC break-even.
- **LV screen (Case B)** additionally *includes* the loss-PV term (branch losses from the full PF).
- **MV/LV screen (Case D)** still *omits* losses **and** any transformer / MV-drop credit — those are
  recovered only implicitly when a zone is fully evacuated and simply not rebuilt (conservative).

**Selection** (`_select_standalone_clusters`, greedy, laminar-safe): compute `avoided` for every
candidate, sort **descending**, and take a candidate iff it is eligible **and**
$\text{NPC}_{\text{SA}} < \text{avoided}$; mark its customers *claimed* so any overlapping (nested,
finer) subtree is skipped. This guarantees no double-counting but is a **greedy heuristic**, not a
global optimum. After selection, the grid is **rebuilt and resized** on the remaining customers
(Case B / D), which re-validates feasibility.

### 3.5 Electrical / economic assumptions embedded here

| Assumption | Consequence |
|---|---|
| Candidate unit = **terminal subtree** | Decisions are per **leaf branch**, never per customer and never the whole zone. Nested laminar family. |
| Decision is **purely economic** | Engineering "remoteness" thresholds are diagnostics only; the gate is $\text{NPC}_{\text{SA}}<\text{avoided}$. |
| **Marginal** avoided cost on the **pre-removal** network | Branch losses/voltages come from the *original* solved PF; the ordering doesn't re-solve after each cut. |
| Case D **omits** losses + transformer/MV credit | Under-credits SHS in the MV/LV screen ⇒ conservative. In practice it no longer restrains anything: the gate accepts **every** candidate branch (22/22 on Mafa) because avoided generation dominates. |
| Service drops **credited and charged** (fixed) | The avoided cost credited avoided service drops while no candidate NPC charged them, so the gate was paid for a saving the objective did not recognise. Both sides now call the same helper on the same metres. |
| Avoided cost **credits avoided grid *generation*** (updated) | Both screens now credit the PV of grid generation LCOE the departing customers no longer draw — the largest previously-missing term. This makes the gate the true break-even and selects more SHS; a residual marginal-vs-joint gap in the *topology* terms remains (see §3.6 Q2). |
| SHS = **catalogue package**, not a dispatch model | No solar resource, battery autonomy, or reliability modelling; NPC via capex+O&M+replacement−salvage. |
| Grid generation priced at **flat 0.25 MU/kWh LCOE** | This term **dominates** total NPC and drives the grid-vs-SHS balance; high sensitivity to the price *level* (SHS_T2 ≈ 0.22/kWh), little to the curve's shape at this price ([`CONVERGENCE_EXPERIMENT.md`](CONVERGENCE_EXPERIMENT.md)). |
| **All-or-nothing** eligibility per subtree | One must-connect / ineligible customer vetoes an otherwise-attractive branch. |

### 3.6 Open questions for the supervisor (Engine 3)

1. **Conservative biases against SHS. (partly addressed)** Avoided grid **generation** LCOE — the
   largest missing term, since generation dominates NPC — is now credited in **both** screens. What
   remains: Case D still omits avoided grid **losses** and any transformer / MV-drop credit, both of
   which would further favour SHS.
2. **Marginal vs. simultaneous. (newly material)** Selection uses avoided costs on the *pre-removal*
   network and a greedy coarsest-first order. With the (exact) generation credit now included, the
   gate is theoretically the true break-even, yet on the example dataset crediting generation raised
   the *realized* Case-B NPC ~1.5% (SHS count 14 → 26): the greedy screen over-credits the
   **topology** terms (branch/pole/loss savings are marginal estimates, but the rebuild re-routes and
   saves less). A re-solve-after-each-cut (commit a cut only if realized NPC drops) or an ILP over
   the laminar family would close this gap — the recommended next step now that generation is in.
3. **Reliability parity.** We compare grid NPC to an SHS *package* NPC with no autonomy/reliability
   term. Is that a fair like-for-like, or do we need an unserved-energy / autonomy-days penalty so a
   grid connection and an SHS deliver comparable service quality before we compare cost?
4. **Generation LCOE sensitivity.** A single flat LCOE (0.25) sets the grid-vs-SHS tipping point and
   dominates NPC. Should this be a resource-/technology-based dispatch cost, and how should we report
   the sensitivity band?
5. **Granularity of the decision.** Leaf-branch (subtree) granularity is coarser than per-customer
   but finer than per-zone. Is the subtree the right electrical/economic unit, or should single
   high-cost **customers** at the ends of a branch be individually screenable?

---

## 4. One-slide synthesis

| | **Engine 1 — LV** | **Engine 2 — MV/transformer** | **Engine 3 — SHS screen** |
|---|---|---|---|
| **Unit built** | Poles, drops, radial LV MST | Zones, transformers, MV MST | Terminal subtrees (leaf branches) |
| **Driving objective** | Min (penalised) **length** | Min LV radius + kVA cap | **Economic**: NPC_SA < avoided |
| **Electrical role** | Reconductoring repairs $\Delta V$/loading | 11 kV evacuation to derated transformers | Carve off branches cheaper off-grid |
| **Key simplification** | Geometry≠electrics; drops impedance-free; balanced 3-φ | Explicit step-up (slack 1.0 pu on LV); `k` by nested search + voltage margin, kVA-capped; centroid siting | Marginal/greedy; omits loss & TX/MV credit (D) |
| **Biggest open question** | Length-MST as a loss/voltage proxy | Flat `cf` at zone scale (needs ADMD); sending voltage/OLTC; *jointly* MV-length-optimal zoning | The flat LCOE: it dominates NPC and the gate now accepts every branch |

**Closing speech line:** *"Each engine is a fast, deterministic geometric heuristic with a thin
electrical validation layer bolted on afterwards. That makes the whole comparison tractable across
four architectures and many settlements — but it means every 'electrical' result is really a
*geometry that survived a power-flow check*, not a geometry chosen to optimise the electrics. The
questions I most want your view on are exactly the seams where that shortcut might bite: the
length-vs-loss topology, the diversity assumption that a single flat coincidence factor has to carry
at both system and zone scale, and — above all — the flat generation LCOE, which is 63–85% of every
NPC, identical between network architectures, and now so dominant that the SHS screen accepts every
branch it is offered."*
