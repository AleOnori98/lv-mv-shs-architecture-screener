# Model Reference — Parameters, Variables & Equations

Precise, component-by-component description of **what the current workflow actually computes**.
Every equation below is taken directly from the implementation; the source module is cited in
each section. Units: power in **kW / kVAr / kVA**, energy in **kWh**, length in **m**, voltage in
**kV** (per-unit `pu` where noted), currency in generic monetary units (MU).

> For a plain-language, step-by-step walkthrough of the same pipeline (what each stage does, the
> data passed between them, the loops and the infeasibility branches), see [`WORKFLOW.md`](WORKFLOW.md).
> For an engine-by-engine methodological discussion and the open questions, see
> [`METHODOLOGY_ENGINES.md`](METHODOLOGY_ENGINES.md). For the known weaknesses and the maturity
> assessment, see [`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md).

Symbols used throughout:

| Symbol | Meaning |
|--------|---------|
| `T` | planning horizon [years] (`planning_horizon_years`) |
| `r` | discount rate [-] (`discount_rate`) |
| `g_i` | annual demand growth of customer `i` (`growth_rate`) |
| `pf_i` | power factor of customer `i` (`power_factor`), clamped to `[0.5, 1.0]` |
| `P_i,now` | present peak active power of customer `i` [kW] (`peak_kw_now`) |
| `cf` | coincidence factor applied to the **design** power flow (`coincidence_factor`) |
| `AF(r,n)` | annuity factor = `(1−(1+r)^−n)/r`, or `n` if `r≤0` |
| `GAF(r,g,n)` | growth annuity factor = `Σ_{t=1..n} (1+g)^(t−1)/(1+r)^t` (`= AF(r,n)` when `g=0`) |

---

## 1. Global conventions

- **Balanced three-phase** steady-state power flow (pandapower `runpp`, flat init, no voltage angles).
- Loads are **peak snapshots** scaled by a **coincidence factor** `cf` — there is no time series in the
  power flow. The design snapshots use the diversified `cf = coincidence_factor` (default **0.6**); an
  extra stress snapshot uses `cf = 1.0`.
- Three demand snapshots are evaluated per candidate:
  - **now** (design): `apply_growth = False`, `cf = coincidence_factor` → `P_i = P_i,now · cf`
  - **horizon** (design): `apply_growth = True`, `cf = coincidence_factor` → `P_i = P_i,now · (1+g_i)^T · cf`
  - **coincident-peak stress**: `apply_growth = True`, `cf = 1.0` → `P_i = P_i,now · (1+g_i)^T`
- **Feasibility** is judged on the **horizon design** snapshot. The **now** snapshot is reported for
  reference; the **coincident-peak stress** snapshot is a robustness check and **does not gate** feasibility.
- **Non-convergence is never a pass.** `run_powerflow` treats a non-converged solve — raised *or*
  reported via `net.converged` — and a converged solve with non-finite bus/branch state as
  infeasible with every metric `NaN`. No limit check is evaluated against `NaN`, because a `NaN`
  comparison is always `False` and would read as "within limits" (§7).

---

## 2. Customer / load model
*Source: `core/electrical/pandapower_builder.py` (`_pq_kw`), `core/io/input_schemas.py`*

Per-customer active and reactive power at a snapshot:

```
P_i      = P_i,now · (1 + g_i)^T · cf     (horizon)   |   P_i = P_i,now · cf   (now)
Q_i      = P_i · tan(arccos(pf_i))                       with pf_i ∈ [0.5, 1.0]
```

Loads are **aggregated to the pole** each customer is associated with:

```
P_pole = Σ_{i ∈ pole} P_i          Q_pole = Σ_{i ∈ pole} Q_i
```

A pandapower `load` is created at each pole bus with `p_mw = P_pole/1000`, `q_mvar = Q_pole/1000`.
Poles with `P_pole ≤ 0` get no load element.

**Customer input fields & defaults** (`normalize_customers`):

| Field | Default if missing | Notes |
|-------|--------------------|-------|
| `peak_kw_now` | `0.3` | kW |
| `annual_kwh_now` | `peak_kw_now · 8760 · 0.2` | i.e. load factor 0.2 |
| `power_factor` | `0.90`, clipped `[0.5,1.0]` | |
| `growth_rate` | `0.03` | per year (compounds over the `T`-year horizon) |
| `demand_tier` | `2` | integer ≥ 0 |
| `priority_weight` | `1.0` | used by topology association ranking |
| `must_connect_flag` | `False` | forces grid service |
| `can_be_standalone_flag` | `True` for households; `False` for grid-priority types¹ | `must_connect ⇒ can_be_standalone=False` |

¹ Grid-priority types: `clinic, hospital, school, productive, water_pump`.
Accepted `customer_type`: `household, shop, productive, school, clinic, hospital, mosque, public_service, water_pump, other`.

---

## 3. Conductors & lines (LV and MV)
*Source: `pandapower_builder._add_line`, `core/electrical/conductor_sizing.py`, `core/io/input_schemas.normalize_conductors`*

Each catalogue conductor carries: `r_ohm_per_km`, `x_ohm_per_km`, `max_i_ka`, `cost_per_m`,
`phase_count` (default 3), `rank` (1 = cheapest/smallest). A line is a pandapower
`create_line_from_parameters` element:

```
length_km     = max(length_m / 1000, 1e-4)
c_nf_per_km   = 0                       (shunt capacitance neglected)
max_i_ka      = conductor.max_i_ka      (thermal ampacity)
```

pandapower then computes per line: current `i_ka`, **`loading_percent = i_ka / max_i_ka · 100`**,
and active loss `pl_mw`.

### 3.1 Iterative conductor sizing
*Source: `conductor_sizing.size_conductors`, `critical_lv_lines`*

Goal: cheapest conductor assignment per line such that the network passes. Algorithm (deterministic):

1. Initialise every line at **rank 0** (cheapest).
2. Build the network, run power flow.
3. If **feasible** → stop (success).
4. Else identify **critical lines**:
   - all **overloaded** lines (`loading_percent > max_line_loading_percent`), **plus**
   - every line on the **shortest path** (weighted by `length_m`) from the slack pole to each
     **undervoltage** bus (`vm_pu < voltage_min_pu`).
   - If no critical line can be isolated → **fallback: upgrade all lines**.
5. Upgrade each critical line by one rank (bounded by the largest available conductor).
6. Repeat until: feasible, or no larger conductor exists anywhere, or `max_iters = 8`.
7. **Early bail:** two consecutive non-convergences (voltage collapse, not fixable by reconductoring)
   stop the loop immediately.

Output: `conductor_by_line`, iteration count, `passed` flag, full per-iteration history.

---

## 4. Poles & service drops + topology engine
*Source: `core/topology/lv_topology.py` → reuses `core.distribution_service._run_low_voltage_core`*

The pole layout, customer→pole association and the LV backbone (MST) are produced by the **reused
legacy engine**. The MVP passes it these engineering parameters:

| Engine argument | MVP parameter | Meaning |
|-----------------|---------------|---------|
| `sampling_distance` | `road_pole_spacing_m` | spacing of candidate poles seeded along roads |
| `user_distance` | `max_user_connection_radius_m` | max customer→pole service-drop radius |
| `max_associations` | `max_users_per_pole` | max customers served by one pole |
| `max_pole_span_m` | `max_pole_span_m` | max LV span between two poles |
| `edge_weight_mode` | `"euclidean"` for LV, `"road_attracted"` for the MV trunk | MST edge weighting |

Derived per-customer quantity used by costs and avoided-cost logic:

```
service_drop_m[i] = distance( building_geometry_i , pole_xy[ pole(i) ] )      [projected CRS]
```

The **slack pole** = pole nearest to the source point (projected). Topology totals:
`total_lv_length_m = Σ edge.length_m`.

Service drops **carry no impedance** in the power flow (loads sit on the pole bus) but they **are
costed**, on both sides of the SHS decision (§10, §10.1).

---

## 5. Transformers (MV/LV)
*Source: `core/topology/mvlv_service_zones.py` (sizing), `pandapower_builder.build_mvlv_net` (model)*

Catalogue fields: `sn_kva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent,
cost_fixed, cost_per_kva, max_loading_percent` (default 100), `rank`.

### 5.1 Zone apparent power
*`zone_apparent_power_kva(ids, customers, T, coincidence_factor)`*

```
cf_tx  = transformer_sizing_coincidence_factor   (default 1.0; 0 ⇒ track coincidence_factor)
P_zone = Σ_{i ∈ zone} P_i,now · (1 + g_i)^T · cf_tx
Q_zone = Σ_{i ∈ zone} P_i · tan(arccos(pf_i))
S_zone = sqrt( P_zone² + Q_zone² )        [kVA]
```

`cf_tx` is a **declared modelling choice**, not an accident, and it is the same basis used for the
zone kVA cap (§11) and the plant step-up (§6):

- At the default **1.0** the rating is sized on the coincident zone peak while the design power flow
  runs at `cf = 0.6`, so the transformer is `1/cf` larger than the snapshot it is validated against
  and its loading peaks near 50% — `max_transformer_loading_percent` cannot bind.
- Setting it to **0** (track the design `cf`) makes the constraint live, but `coincidence_factor` is a
  single flat **system-wide** number and diversity only emerges with many aggregated customers, so
  applied to a 10-customer zone it understates that zone's real peak and under-sizes the small zones.
- It is also not cost-free: catalogue impedance is quoted on the unit's own rating, so a **smaller
  unit has a higher ohmic impedance** (`vk% / sn`). Measured on Mafa, tracking the design factor costs
  ~0.010 pu of LV voltage and pushed Case D from feasible (8 zones, 0.916 pu) to infeasible
  (10 zones, 0.8998 pu). The conservatism was partly buying voltage support.

The principled resolution is a **size-dependent ADMD / diversity curve** (`cf` as a function of the
number of aggregated customers) rather than either flat number; until then the basis stays explicit,
single-sourced and reported.

### 5.2 Transformer selection
```
target = S_zone · TRANSFORMER_SIZING_SAFETY_FACTOR      (= 1.2)
choose the smallest-rank transformer with:  sn_kva · max_loading_percent/100 ≥ target
if none qualifies → choose the largest available transformer
```

The same 1.2 also divides the zone kVA cap (§11). That is **one margin used consistently**, not a
compounded 1.44: the cap exists precisely so a zone sitting at the cap is still coverable by the
largest catalogue unit.

### 5.3 pandapower model
`create_transformer_from_parameters` with `sn_mva = sn_kva/1000` and the catalogue impedances
(`vk_percent, vkr_percent, pfe_kw, i0_percent`). The LV winding bus is tied to the zone's slack
pole by a closed bus–bus switch. pandapower reports transformer **`loading_percent`**; feasibility
requires `≤ max_transformer_loading_percent`.

---

## 6. Source / slack
*Source: `pandapower_builder`*

- **Flat LV networks (A, B):** one `ext_grid` at the slack pole, `vm_pu = 1.0`,
  `vn_kv = lv_voltage_kv` (0.4). There is **no plant transformer** on the LV cases — an ideal
  0.4 kV slack. This is an asymmetry against C/D, which do pay for a step-up (see
  `PIPELINE_REVIEW.md` §3.5).
- **MV/LV networks (C, D):** a **single** slack on the **plant LV bus** at `source_vm_pu`
  (default **1.0**, the honest sending voltage), an **explicit step-up transformer** (LV→MV) feeding
  the MV backbone, and a step-down at each zone. A single-slack invariant is asserted after build.
  The step-up is sized by `build_stepup_spec` on the settlement's apparent power at `cf_tx` × 1.2,
  may exceed the catalogue (a project-procured unit), and its CAPEX enters `transformer_cost`.
  No OLTC and no explicit upstream source impedance are modelled.

---

## 7. Power flow & feasibility
*Source: `core/electrical/validation.py`*

After `runpp`:

```
min_vm_pu = min(res_bus.vm_pu)      max_vm_pu = max(res_bus.vm_pu)
max_line_loading = max(res_line.loading_percent)
max_trafo_loading = max(res_trafo.loading_percent)
losses_kw   = (Σ res_line.pl_mw + Σ res_trafo.pl_mw) · 1000
losses_%    = 100 · losses_kw / total_load_kw
```

**Feasible** iff power flow **converged** AND all of:

```
min_vm_pu ≥ voltage_min_pu
max_vm_pu ≤ voltage_max_pu
max_line_loading ≤ max_line_loading_percent
max_trafo_loading ≤ max_transformer_loading_percent
```

A non-converged solve returns every metric as `NaN` with bottleneck
`"power flow did not converge"`; **no limit check runs**, and `losses_kw = NaN` propagates into the
cost model as an *unknown* (not zero) component (§10).

`violation_severity(outcome, constraints)` scores how far a converged outcome sits outside its
limits, in comparable units (voltage breaches in pu, loading breaches as a fraction of the limit);
`inf` for a non-converged outcome. It orders **infeasible** candidates, where NPC is not a
meaningful ranking (§12).

**Coincident-peak stress outcome** (`outcome_stress`): the same metrics on the `cf = 1.0` horizon
snapshot, on the network already sized for the diversified design peak. Stored per candidate and
surfaced in the dashboard (`coincident_*` columns, a second headroom bar, a map scenario) but
**informational only** — it never changes which candidate is feasible or recommended.

---

## 8. SHS candidate detection — terminal subtrees (leaf branches)
*Source: `core/topology/subtree_candidates.py`, `core/topology/branch_detection.py`*

SHS candidates are the **terminal subtrees** of a rooted radial LV tree. The same builder serves the
Full-LV tree (Case B) and **each MV/LV zone's** LV tree (Case D).

**Terminal-subtree generation** (`terminal_subtrees`): root the pole graph at the slack pole; a
**fork** is any node with ≥ 2 children. For each fork `B` and each child `c`, the candidate subtree
is every pole at/below `c`, and the single edge `(B, c)` is its **entry edge**. Consequences: the
un-cuttable trunk and the whole zone are **never** candidates; subtrees **nest** and form a
**laminar** family (any two are nested or disjoint); a pure chain with no fork yields no candidate.

Per-subtree metrics (`subtree_clusters`):

```
branch_edge_ids          = internal edges + entry edge
branch_length_m          = Σ edge.length_m over branch_edge_ids
branch_cost              = Σ edge.length_m · cost_per_m · civil_logistics_multiplier
branch_cost_per_customer = branch_cost / n_customers
service_drop_m           = Σ service-drop length of the subtree's customers          (metadata)
branch_losses_kw         = Σ pandapower line losses on branch_edge_ids
                           (Case B: from the Full-LV PF; zone candidates: 0 — no per-zone PF slice)
min_voltage, max_line_loading = sliced from the solved PF when supplied (Case B); NaN for zones
uses_largest_conductor   = any branch line on the largest-rank LV conductor
peak_kw_horizon          = Σ P_i,now · (1 + g_i)^T
```

**Diagnostic flags only — these never gate selection** (`_score`): high `branch_cost_per_customer`,
undervoltage, largest-conductor use. *(The former `dmax`/`lmax` distance & path-length flags have
been removed entirely.)*

**Stand-alone precheck** (subtree level): `all(¬must_connect_flag) AND all(can_be_standalone_flag)`.

---

## 9. Stand-alone (solar) systems
*Source: `core/economics/standalone.py`, `core/economics/npc.py`*

### 9.1 Eligibility (per customer)
```
must_connect_flag      = False
can_be_standalone_flag = True
tol    = 1 + standalone_sizing_tolerance              (default 1.02)
growth = (1 + g_i)^T                                  (size_on_horizon, applied to BOTH peak and energy)
∃ package with:  service_tier          ≥ demand_tier
                 max_peak_kw  · tol    ≥ peak_kw_now     · growth
                 annual_kwh   · tol    ≥ annual_kwh_now  · growth
```
A **candidate subtree** is stand-alone-eligible only if **every** customer in it is eligible.

`standalone_sizing_tolerance` exists because the catalogue is discrete and the horizon peak is a
compounded number, which makes a zero-tolerance test a knife edge: the median Mafa household
(0.25 kW, `g=0.04`, `T=15`) needs **0.45024 kW** against `SHS_T2`'s **0.45 kW** and misses by
**0.05%**, which pushes it to `SHS_T3` at an NPC of 1329 instead of 823 (+61%). With all-or-nothing
subtree eligibility, one such customer also vetoes an entire branch. Across Mafa's 227 eligible
customers the package mix moves from `{T3: 204, T4: 23}` to `{T2: 125, T3: 79, T4: 23}` and the SHS
NPC from 339k to 275k.

The energy test applies the **same** horizon growth as the peak test; previously a horizon-sized
peak was compared against an ungrown `annual_kwh_now`, so packages could qualify on energy they
would be short of by year `T`.

### 9.2 Package net present cost (NPC)
For a chosen package over horizon `T` at rate `r`:

```
om_annual  = annual_om_fraction · capex
repl_cost  = replacement_fraction · capex
NPC_pkg    = capex
           + om_annual · AF(r, T)                                   (PV of O&M)
           + Σ_{t = life, 2·life, … , t < T}  repl_cost / (1+r)^t   (PV of replacements)
           − salvage / (1+r)^T                                      (PV of salvage credit)

salvage    = residual · repl_cost,   residual = (last_install + life − T)/life  if > 0 else 0
```

The cheapest eligible package (min `NPC_pkg`) is chosen per customer; a cluster's stand-alone NPC
= Σ per-customer NPC. There is **no reliability / autonomy term**: an SHS package and a grid
connection are compared on cost alone (see `METHODOLOGY_ENGINES.md` §3.6 Q3).

---

## 10. Cost model & candidate NPC
*Source: `core/economics/cost_model.py`*

```
lv_line_cost          = ( Σ_lines length_m · cost_per_m ) · civil_logistics_multiplier
mv_line_cost          = ( Σ_mv-lines length_m · cost_per_m ) · civil_logistics_multiplier
pole_cost             = n_poles · pole_cost · civil_logistics_multiplier
transformer_cost      = Σ_tx ( cost_fixed + cost_per_kva · sn_kva )      (incl. the plant step-up)
connection_metering   = n_grid_customers · ( lv_connection_cost_per_customer
                                             + metering_protection_cost_per_customer )
service_drop_cost     = Σ service_drop_m · cheapest_LV_cost_per_m · civil_logistics_multiplier

g_set                 = energy-weighted mean growth of the candidate's grid customers
g_loss                = (1 + g_set)² − 1                 (losses scale with current squared)

loss_pv_cost          = ( losses_kw / (1+g_loss)^T )     (back-cast the HORIZON snapshot to year 0)
                        · 8760 · loss_load_factor         (annual loss energy, kWh)
                        · loss_energy_price               (→ annual cost)
                        · GAF(r, g_loss, T)               (→ PV over the growth trajectory)

generation_npc        = Σ_{grid customers i}  annual_kwh_now_i
                        · generation_lcoe_per_kwh
                        · GAF(r, g_i, T)                  (each customer's own growth)

grid_capex   = lv_line_cost + mv_line_cost + pole_cost + transformer_cost
               + connection_metering + service_drop_cost
total_npc    = grid_capex + standalone_npc + loss_pv_cost + generation_npc
```

**`loss_energy_price`** = `loss_value_per_kwh` when set > 0, else `generation_lcoe_per_kwh`. Loss
energy is *generated* energy that never reaches a customer, so by default it carries the same price
as every other generated kWh; pricing it separately put two prices on one commodity. Together the
two energy components cost **generation**, not delivery.

**Growth consistency.** Both energy streams are levelised with `GAF` over their actual trajectory.
Previously generation held **present-year** energy flat for the whole horizon while the loss cost was
taken from the **horizon** power flow, also held flat — two different years of the same growing
demand summed into one NPC. Measured on Mafa: generation 1.011M → 1.234M (flat-at-present
under-counted) and loss PV 87.6k → 44.8k (flat-at-horizon over-counted); the two old bases bracketed
the right answer from opposite sides.

**Unknown components.** `loss_present_value` returns **`NaN`** when the loss snapshot is unknown
(diverged power flow) — *not* 0.0. `CostBreakdown` exposes `missing_components` and `npc_complete`;
`total_npc` sums the **known** components and is a **lower bound** whenever `npc_complete` is false.
Charging zero for the losses of an unsolvable network previously made every diverged candidate the
cheapest one.

### 10.1 Avoided grid cost & SHS selection (Cases B and D)
*Source: `mvp_runner._avoided_grid_cost`, `_avoided_lv_cost`, `_avoided_mvlv_subtree_cost`, `_select_standalone_clusters`*

Marginal grid cost avoided if a candidate subtree's customers all move to stand-alone:

```
avoided = branch_cost                                          (internal lines + entry edge)
        + service_drop_cost(subtree drop metres)               (same helper the NPC charges)
        + connection_metering(n_subtree)
        + pole_cost(n_subtree_poles)        (ALL subtree poles; the boundary pole is upstream, excluded)
        + generation_npc(subtree customers)                    (avoided grid generation, BOTH screens)
        [ + loss_pv_cost(branch_losses_kw, subtree customers) ]            (LV screen only)
```

Every term uses the **same helper** as the corresponding NPC component, so the gate cannot be paid
for a saving the objective does not recognise. (Service drops were previously credited here while no
candidate's NPC charged them.)

- **LV screen** (`_avoided_lv_cost`, Case B) **includes** the loss-PV term (branch losses from the
  Full-LV power flow).
- **MV/LV screen** (`_avoided_mvlv_subtree_cost`, Case D) **omits** losses **and** the zone
  transformer / MV service drop — those savings are credited only implicitly, when a zone empties
  completely and is not rebuilt (deliberately conservative).

**Selection** (`_select_standalone_clusters`, purely economic + greedy):

```
eligible candidate  ⇔  stand-alone precheck passes AND every customer has a feasible package
decision            ⇔  SHS_NPC < avoided
order               = descending avoided cost; a customer already claimed by a selected (larger,
                      nested) subtree blocks any overlapping candidate → no double counting (laminar)
```

> **Known limitation.** The gate currently accepts **every** candidate branch on the real case
> studies (17/17 in Case B, 22/22 in Case D on Mafa). Avoided *generation* dominates the avoided
> cost, and at 0.25 MU/kWh the grid price sits above what SHS_T2 costs per kWh (≈ 0.22 levelised),
> with no service-quality term to separate the two. The flat LCOE's lack of scale economies is
> **not** the cause: realistic synthetic curves leave the Mafa answer unchanged, and the coupled
> loop converges in 2–3 rounds. See [`CONVERGENCE_EXPERIMENT.md`](CONVERGENCE_EXPERIMENT.md).

Avoided costs are **marginal estimates on the pre-removal network**, while the realised candidate is
a full rebuild (Case D even re-partitions from scratch), so realised savings differ from credited
ones.

---

## 11. MV/LV service-zone construction
*Source: `core/topology/mvlv_service_zones.py`, `mvp_runner._build_mvlv`, `_voltage_cap_search`*

The whole grid-connected set is partitioned into **capacity-limited zones**. The zone **count `k`**
is set by `clustering_criterion`; the **kVA cap** is enforced under all three criteria by
`enforce_kva_cap`, which recursively bisects any zone over the cap:

```
max_zone_kva = max_t( sn_kva_t · max_loading_percent_t / 100 ) / TRANSFORMER_SIZING_SAFETY_FACTOR
```

**(a) `voltage_cap` (default) — power-flow-driven.** The partition family is a **Ward dendrogram cut
at `k`** (`partition_hierarchical_k`), so the `k+1` partition **refines** the `k` partition: a zone is
split, never reshuffled. This matters because the previous k-means-per-`k` re-solved from scratch, so
the worst voltage drop was not even monotone in `k` (measured on Mafa: `k=4` at 0.1028 pu worse than
`k=3` at 0.1004 pu) — a search that stops at the first failure cannot rest on that.

A partition is **accepted** at `min_vm_pu ≥ voltage_min_pu + design_voltage_margin_pu`, not merely at
the feasibility limit, because minimising `k` against the bare limit spends every volt of headroom
and leaves the chosen design sitting on the constraint (0.913 pu against a 0.90 limit, measured).

The margin is a **preference, not a constraint**. Adding transformers shortens LV feeders but
lengthens the MV trunk and inserts another transformer impedance, so the achievable voltage
**plateaus** — on Mafa no zone count reaches a 0.92 pu target. When the margin is unreachable the
search falls back to the **smallest merely-feasible** count and records why; if nothing is feasible it
reports the least-violating partition. Search bounds:

- `k` never exceeds the customer count, and a partition failure is recorded rather than fatal;
- diverged attempts step **geometrically** (divergence means grossly too few zones, not one short);
- the climb stops once extra transformers stop buying voltage — but **only after** something feasible
  is in hand, because on Mafa the voltage sits flat at 0.884 pu for `k=4..7` and then *jumps* to
  0.905 at `k=10`: flat is not saturated.

Setting `k_selection_snapshot = "coincident"` selects against the coincident-peak snapshot instead.
Candidate feasibility itself is unaffected by the margin — it governs which `k` the *designer* picks.

**(b) `distance_cap` — geometric guarantee, one-shot.** Complete-linkage clustering cut at
`max_cluster_diameter_m`, so every cluster diameter is ≤ the cap by construction; `k` emerges from the
cut, no search and no power flow. Errors if the cap needs `k >` `max_transformers`.

**(c) `capacity` — legacy greedy farthest-first.** Seed = unassigned customer farthest from the
source; grow by nearest customers while `dist ≤ max_lv_zone_radius_m`,
`members < max_zone_customers` and `Σ|S| ≤ max_zone_kva`. Here `k` is a byproduct of packing.

> `max_lv_zone_radius_m` and `max_zone_customers` are read **only** by the `capacity` criterion. They
> are ignored under the default `voltage_cap` and under `distance_cap`.

Each zone → transformer at the **unweighted centroid**, snapped to the nearest road (not the
kVA-weighted load centroid); a local LV tree (§4) → its own sizing loop (§3.1). The MV trunk is an
MST over `{source} ∪ {transformers}` routed by the same engine (road-attracted when roads exist);
the MV conductor is **one type for the whole trunk**, swept by rank until the full network passes.

---

## 12. Candidate assembly & recommendation
*Source: `core/workflow/mvp_runner.py`*

Orchestration order: **A** → **C** → LV SHS screen → **B** → MV/LV SHS screen → **D** (the two
screens are independent; C is built before D and never reuses B's selection).

| Candidate | Construction |
|-----------|--------------|
| **A — Full LV** | all customers on one flat LV network, conductors sized (§3.1) |
| **B — LV + Stand-alone** | LV terminal-subtree screen (§8, §10.1); LV rebuilt + resized on the remainder |
| **C — MV/LV** | all grid customers partitioned into transformer zones (§11); per-zone LV + MV trunk sized |
| **D — MV/LV + Stand-alone** | its **own** per-zone subtree screen over Case C's zones; the remainder re-partitioned and MV/LV-rebuilt (D = C if none selected) |

> Cases C and D are built over **Case A's** `grid_customer_ids`, so a customer the flat-LV topology
> engine could not associate to a pole can never be served by MV/LV either (`PIPELINE_REVIEW.md` §3.8).

**Service-obligation checks** (per candidate):
```
must_connect_violations = | must_connect_customers − grid_customers |   ( > 0 ⇒ infeasible )
service_tier_violations = | all_customers − (grid ∪ standalone) |        ( reported )
```

**Recommendation:**
```
if any feasible:   recommended = argmin_{feasible} total_npc
else:              recommended = argmin_{all} _infeasibility_rank
                   where _infeasibility_rank = ( horizon PF converged?,
                                                 must_connect_violations,
                                                 service_tier_violations,
                                                 violation_severity,
                                                 npc_complete?,
                                                 total_npc )
```

The infeasible fallback does **not** rank by NPC. A diverged candidate has no computable loss cost,
so its total is a lower bound and it would win the comparison precisely *because* it is broken.

---

## 13. Default parameter tables

### 13.1 Cost parameters (`COST_PARAMETER_DEFAULTS`, fills any missing key)

| Parameter | Default | Parameter | Default |
|-----------|---------|-----------|---------|
| `discount_rate` | 0.08 | `voltage_min_pu` | 0.90 |
| `planning_horizon_years` | 15 | `voltage_max_pu` | 1.05 |
| `lv_connection_cost_per_customer` | 100 | `max_line_loading_percent` | 100 |
| `metering_protection_cost_per_customer` | 50 | `max_transformer_loading_percent` | 100 |
| `pole_cost` | 120 | `max_lv_zone_radius_m` | 250 |
| `civil_logistics_multiplier` | 1.2 | `max_zone_customers` | 60 |
| `loss_value_per_kwh` | 0.20 | `clustering_criterion` | `voltage_cap` |
| `loss_load_factor` | 0.30 | `max_cluster_diameter_m` | 500 |
| `generation_lcoe_per_kwh` | 0.25 | `max_transformers` | 12 |
| `coincidence_factor` | 0.6 | `seed_diameter_m` | 2000 |
| `source_vm_pu` | 1.00 | `design_voltage_margin_pu` | 0.02 |
| `branch_cost_per_customer_threshold` | 600 | `k_selection_snapshot` | `design` |
| `standalone_sizing_tolerance` | 0.02 | `transformer_sizing_coincidence_factor` | 1.0 |

`generation_lcoe_per_kwh` (EUR/kWh) prices the mini-grid generation of grid-served energy (§10) and,
by default, its losses too. It is **63–85% of `total_npc`** on every candidate of every bundled case
study and is **identical between network architectures** (A and C agree to the euro), so it cannot
discriminate LV from MV/LV while completely dominating the grid-vs-SHS balance. No bundled dataset
sets it — every published number rests on this default. See `PIPELINE_REVIEW.md` §3.1.

> **Retired** (accepted silently from older input files, ignored by the engine):
> `dmax_lv_service_zone_m`, `lmax_lv_path_m`, `remote_branch_length_threshold_m`,
> `loss_threshold_percent` — SHS detection is threshold-free (§8).
> `branch_cost_per_customer_threshold` is used **only** as a diagnostic flag, never as a gate.
> Any **other** unrecognised key now raises a warning with a did-you-mean suggestion instead of
> being dropped in silence (`normalize_cost_parameters(..., strict=True)` makes it an error).

### 13.2 Fixed engineering coefficients (in code, not user inputs)

| Coefficient | Value | Where |
|-------------|-------|-------|
| Transformer sizing safety factor | 1.2 | `TRANSFORMER_SIZING_SAFETY_FACTOR` (sizing target **and** zone cap divisor) |
| Step-up sizing margin | 1.2 | `build_stepup_spec` |
| Flat-LV source slack voltage | 1.00 pu | `build_lv_net` (ignores `source_vm_pu`) |
| Power-factor clamp | [0.5, 1.0] | `_pq_kw` |
| Shunt capacitance `c_nf_per_km` | 0 | `_add_line` |
| Sizing max iterations | 8 | `size_conductors` |
| Non-convergence bail streak | 2 | `size_conductors` |
| k-search plateau patience / epsilon | 3 attempts / 0.002 pu | `_voltage_has_plateaued` |
| Hours per year | 8760 | loss energy |

> **Modelling limitations:** balanced 3-phase only; single fixed source; heuristic road-pole + MST
> topology (min length, not min loss); impedance-free service drops; leaf-branch-level (not
> per-customer) stand-alone decisions; heuristic zoning and centroid transformer placement; one MV
> conductor for the whole trunk; no reliability/autonomy term; no hourly dispatch behind the LCOE;
> illustrative cost values. See `PIPELINE_REVIEW.md` for the full assessment.
