# Model Reference — Parameters, Variables & Equations

Precise, component-by-component description of **what the current workflow actually computes**.
Every equation below is taken directly from the implementation; the source module is cited in
each section. Units: power in **kW / kVAr / kVA**, energy in **kWh**, length in **m**, voltage in
**kV** (per-unit `pu` where noted), currency in generic monetary units (MU).

> For a plain-language, step-by-step walkthrough of the same pipeline (what each stage does, the
> data passed between them, the loops and the infeasibility branches), see [`WORKFLOW.md`](WORKFLOW.md).

Symbols used throughout:

| Symbol | Meaning |
|--------|---------|
| `T` | planning horizon [years] (`planning_horizon_years`) |
| `r` | discount rate [-] (`discount_rate`) |
| `g_i` | annual demand growth of customer `i` (`growth_rate`) |
| `pf_i` | power factor of customer `i` (`power_factor`), clamped to `[0.5, 1.0]` |
| `P_i,now` | present peak active power of customer `i` [kW] (`peak_kw_now`) |
| `AF(r,n)` | annuity factor = `(1−(1+r)^−n)/r`, or `n` if `r≤0` |

---

## 1. Global conventions

- **Balanced three-phase** steady-state power flow (pandapower `runpp`, flat init, no voltage angles).
- Loads are **peak snapshots** scaled by a **coincidence factor** `cf` — there is no time series in the
  power flow. The design snapshots use the diversified `cf = coincidence_factor` (default **0.6**); an
  extra stress snapshot uses `cf = 1.0` (every customer at their own peak simultaneously).
- Three demand snapshots are evaluated per candidate:
  - **now** (design): `apply_growth = False`, `cf = coincidence_factor` → `P_i = P_i,now · cf`
  - **horizon** (design): `apply_growth = True`, `cf = coincidence_factor` → `P_i = P_i,now · (1 + g_i)^T · cf`
  - **coincident-peak stress**: `apply_growth = True`, `cf = 1.0` → `P_i = P_i,now · (1 + g_i)^T`
- **Feasibility** is judged on the **horizon design** snapshot. The **now** snapshot is reported for
  reference; the **coincident-peak stress** snapshot is a robustness check and **does not gate** feasibility.

---

## 2. Customer / load model
*Source: `core/electrical/pandapower_builder.py` (`_pq_kw`), `core/io/input_schemas.py`*

Per-customer active and reactive power at a snapshot:

```
P_i      = P_i,now · (1 + g_i)^T · cf     (horizon)   |   P_i = P_i,now · cf   (now)
Q_i      = P_i · tan(arccos(pf_i))                       with pf_i ∈ [0.5, 1.0]
           cf = coincidence_factor for the design snapshots (default 0.6);  cf = 1.0 for the coincident-peak stress
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
r_ohm_per_km  = conductor.r_ohm_per_km
x_ohm_per_km  = conductor.x_ohm_per_km
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
7. **Early bail:** if the power flow fails to converge **2 iterations in a row** (voltage collapse,
   not fixable by reconductoring) the loop stops immediately.

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
| `edge_weight_mode` | `"euclidean"` (fixed) | MST edge weighting (roads still seed pole positions) |

Derived per-customer quantity used by costs and avoided-cost logic:

```
service_drop_m[i] = distance( building_geometry_i , pole_xy[ pole(i) ] )      [projected CRS]
```

The **slack pole** = pole nearest to the source point (projected). Topology totals:

```
total_lv_length_m = Σ edge.length_m
```

---

## 5. Transformers (MV/LV)
*Source: `core/topology/mvlv_service_zones.py` (sizing), `pandapower_builder.build_mvlv_net` (model)*

Catalogue fields: `sn_kva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent,
cost_fixed, cost_per_kva, max_loading_percent` (default 100), `rank`.

### 5.1 Zone apparent power (horizon, coincident)
Zoning and transformer sizing deliberately use the **coincident** horizon peak (`cf = 1.0`, no diversity),
so transformers are rated for the worst case even though the LV design power flow runs at the diversified
peak (`cf = coincidence_factor`). This is conservative on purpose.
```
P_zone = Σ_{i ∈ zone} P_i,now · (1 + g_i)^T
Q_zone = Σ_{i ∈ zone} P_i · tan(arccos(pf_i))
S_zone = sqrt( P_zone² + Q_zone² )        [kVA]
```

### 5.2 Transformer selection
```
target = S_zone · safety_factor            (safety_factor = 1.2)
choose the smallest-rank transformer with:  sn_kva · max_loading_percent/100 ≥ target
if none qualifies → choose the largest available transformer
```

### 5.3 pandapower model
`create_transformer_from_parameters` with `sn_mva = sn_kva/1000` and the catalogue impedances
(`vk_percent, vkr_percent, pfe_kw, i0_percent`). The LV winding bus is tied to the zone's slack
pole by a closed bus–bus switch. pandapower reports transformer **`loading_percent`**; feasibility
requires `≤ max_transformer_loading_percent`.

---

## 6. Source / slack
*Source: `pandapower_builder`*

- **Flat LV networks (A, B):** one `ext_grid` at the slack pole, `vm_pu = 1.0`, `vn_kv = lv_voltage_kv` (0.4).
- **MV/LV networks (C, D):** source/slack set at **`source_vm_pu = 1.04`** (realistic high sending
  voltage to offset MV-line + transformer drop). MV buses at `mv_voltage_kv` (11), LV buses at 0.4.
  The MV island and any LV core are each referenced to the source as slacks — **no explicit step-up
  transformer is modelled** (documented MVP simplification).

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

Non-convergence ⇒ infeasible with bottleneck `"power flow did not converge"`. Otherwise the
`active_bottleneck` lists every violated limit.

**Coincident-peak stress outcome** (`outcome_stress`): the same metrics recomputed on the coincident-peak
snapshot (`cf = 1.0`, horizon — every customer at their own peak at once) on the network already sized for
the diversified design peak. It is stored per candidate and surfaced in the dashboard (comparison columns
`coincident_*`, a second headroom bar, the map's "coincident peak" scenario) but is **informational only**:
it never changes which candidate is feasible or recommended. A candidate can pass the design snapshot yet
breach limits — or fail to converge — under the coincident peak, which is exactly the robustness signal.

---

## 8. SHS candidate detection — terminal subtrees (leaf branches)
*Source: `core/topology/subtree_candidates.py`, `core/topology/branch_detection.py`*

SHS candidates are the **terminal subtrees** of a rooted radial LV tree — not threshold-defined
"remote clusters". The same builder serves the Full-LV tree (Case B) and **each MV/LV zone's** LV
tree (Case D).

**Terminal-subtree generation** (`terminal_subtrees`): root the pole graph at the slack pole; a
**fork** is any node with ≥ 2 children. For each fork `B` and each child `c`, the candidate subtree
is every pole at/below `c` (toward the leaves), and the single edge `(B, c)` is its **entry edge**.
Consequences: the un-cuttable trunk and the whole zone are **never** candidates; subtrees **nest**
(a hamlet contains its stubs) and form a **laminar** family (any two are nested or disjoint); a pure
chain with no fork yields no candidate.

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
∃ package with:  service_tier ≥ demand_tier
                 max_peak_kw  ≥ peak           (peak = P_now·(1+g)^T, horizon-sized)
                 annual_kwh   ≥ annual_kwh_now
```
A **candidate subtree** is stand-alone-eligible only if **every** customer in it is eligible.

### 9.2 Package net present cost (NPC)
For a chosen package over horizon `T` at rate `r`:

```
om_annual  = annual_om_fraction · capex
repl_cost  = replacement_fraction · capex          (default replacement_fraction = 0.8)
NPC_pkg    = capex
           + om_annual · AF(r, T)                                   (PV of O&M)
           + Σ_{t = life, 2·life, … , t < T}  repl_cost / (1+r)^t   (PV of replacements)
           − salvage / (1+r)^T                                      (PV of salvage credit)

salvage    = residual · repl_cost,   residual = (last_install + life − T)/life  if > 0 else 0
```

Replacements occur at each **full lifetime** strictly before `T`. The cheapest eligible package
(min `NPC_pkg`) is chosen per customer; a cluster's stand-alone NPC = Σ per-customer NPC.

---

## 10. Cost model & candidate NPC
*Source: `core/economics/cost_model.py`*

```
lv_line_cost          = ( Σ_lines length_m · cost_per_m ) · civil_logistics_multiplier
mv_line_cost          = ( Σ_mv-lines length_m · cost_per_m ) · civil_logistics_multiplier
pole_cost             = n_poles · pole_cost · civil_logistics_multiplier
transformer_cost      = Σ_tx ( cost_fixed + cost_per_kva · sn_kva )
connection_metering   = n_grid_customers · ( lv_connection_cost_per_customer
                                             + metering_protection_cost_per_customer )

loss_pv_cost          = losses_kw · 8760 · loss_load_factor          (annual loss energy, kWh)
                        · loss_value_per_kwh                          (→ annual cost)
                        · AF(r, T)                                    (→ present value)

generation_npc        = ( Σ_{grid customers} annual_kwh_now )        (grid-served annual energy, kWh)
                        · generation_lcoe_per_kwh                     (→ annual generation cost)
                        · AF(r, T)                                    (→ present value)

grid_capex   = lv_line_cost + mv_line_cost + pole_cost + transformer_cost + connection_metering
total_npc    = grid_capex + standalone_npc + loss_pv_cost + generation_npc
```

`generation_npc` levelises the **mini-grid generation cost** for grid-served energy. Stand-alone customers
are **excluded** (their generation is already inside `standalone_npc` via the SHS package), so grid and
stand-alone are compared on the same footing — without it the grid architectures looked artificially cheap.
It is applied to `annual_kwh_now` (present energy, held flat and PV'd over the horizon like `loss_pv_cost`).

### 10.1 Avoided grid cost & SHS selection (Cases B and D)
*Source: `mvp_runner._avoided_grid_cost`, `_avoided_lv_cost`, `_avoided_mvlv_subtree_cost`, `_select_standalone_clusters`*

Marginal grid cost avoided if a candidate subtree's customers all move to stand-alone:

```
avoided = branch_cost                                          (internal lines + entry edge)
        + service_drop_m · cheapest_LV_cost_per_m · civil_logistics_multiplier
        + connection_metering(n_subtree)
        + pole_cost(n_subtree_poles)        (ALL subtree poles; the boundary pole is upstream, excluded)
        [ + loss_present_value(branch_losses_kw) ]            (LV screen only)
```

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

Service drops are priced with the cheapest LV conductor (a conservative proxy; they are not separate
pandapower lines). The §8 thresholds are reported as diagnostics only and never gate this decision.

---

## 11. MV/LV service-zone construction
*Source: `core/topology/mvlv_service_zones.py`, `mvp_runner._build_mvlv`*

Whole grid-connected set is partitioned into **capacity-limited zones** (greedy farthest-first,
deterministic):

```
max_zone_kva = max_t( sn_kva_t · max_loading_percent_t / 100 ) / 1.2        (largest TX, derated)

seed         = unassigned customer farthest from the source
grow zone by nearest customers c subject to:
        dist(c, seed) ≤ max_lv_zone_radius_m
        members < max_zone_customers
        Σ |S_c| + |S_new| ≤ max_zone_kva
repeat until all customers assigned
```

Each zone → transformer at the zone centroid (snapped to nearest road); a local LV tree (engine);
MV trunk = MST over `{source} ∪ {transformers}`. LV conductors are sized per zone (§3.1); the MV
conductor is upgraded by rank until the full MV/LV network passes.

---

## 12. Candidate assembly & recommendation
*Source: `core/workflow/mvp_runner.py`*

Orchestration order: **A** → **C** → LV SHS screen → **B** → MV/LV SHS screen → **D** (the two screens
are independent; C is built before D and never reuses B's selection).

| Candidate | Construction |
|-----------|--------------|
| **A — Full LV** | all customers on one flat LV network, conductors sized (§3.1) |
| **B — LV + Stand-alone** | LV terminal-subtree screen (§8, §10.1): leaf branches moved off-grid where `NPC_SA < avoided`; LV rebuilt + resized on the remainder |
| **C — MV/LV** | all grid customers partitioned into transformer zones (§11); per-zone LV + MV trunk sized |
| **D — MV/LV + Stand-alone** | its **own** per-zone terminal-subtree screen (§8, §10.1) over Case C's zones; selected leaf branches removed and the remainder MV/LV-rebuilt (D = C if none selected) |

**Service-obligation checks** (per candidate):
```
must_connect_violations = | must_connect_customers − grid_customers |   ( > 0 ⇒ infeasible )
service_tier_violations = | all_customers − (grid ∪ standalone) |        ( reported )
```

**Recommendation:**
```
if any feasible:   recommended = argmin_{feasible} total_npc
else:              recommended = argmin_{all} total_npc   (least-infeasible, bottleneck reported)
```

---

## 13. Default parameter tables

### 13.1 Cost parameters (`COST_PARAMETER_DEFAULTS`, fills any missing key)

| Parameter | Default | Parameter | Default |
|-----------|---------|-----------|---------|
| `discount_rate` | 0.08 | `voltage_min_pu` | 0.90 |
| `planning_horizon_years` | 15 | `voltage_max_pu` | 1.05 |
| `lv_connection_cost_per_customer` | 100 | `max_line_loading_percent` | 100 |
| `metering_protection_cost_per_customer` | 50 | `max_transformer_loading_percent` | 100 |
| `pole_cost` | 120 | `dmax_lv_service_zone_m` | 800 |
| `civil_logistics_multiplier` | 1.2 | `lmax_lv_path_m` | 1000 |
| `loss_value_per_kwh` | 0.20 | `remote_branch_length_threshold_m` | 300 |
| `loss_load_factor` | 0.30 | `branch_cost_per_customer_threshold` | 600 |
| `generation_lcoe_per_kwh` | 0.25 | `loss_threshold_percent` | 8 |
| `coincidence_factor` | 0.6 | `max_lv_zone_radius_m` | 250 |
| `max_zone_customers` | 60 | | |

`generation_lcoe_per_kwh` (EUR/kWh) prices the mini-grid generation of grid-served energy (§10);
`coincidence_factor` is the diversity applied to the **design** power flow (§1–2), with the coincident-peak
stress test fixed at 1.0. Both are also exposed as inputs in the page's Section 2.

> **No longer used by the engine** (kept in defaults for backward compatibility):
> `dmax_lv_service_zone_m`, `lmax_lv_path_m`, `remote_branch_length_threshold_m`, `loss_threshold_percent`
> — SHS detection is now threshold-free (§8). `branch_cost_per_customer_threshold` is used **only** as a
> diagnostic flag, never as a gate.

### 13.2 Fixed engineering coefficients (in code, not user inputs)

| Coefficient | Value | Where |
|-------------|-------|-------|
| Transformer sizing safety factor | 1.2 | `select_transformer` |
| Zone-capacity derating divisor | 1.2 | `_build_mvlv` |
| MV/LV source slack voltage | 1.04 pu | `build_mvlv_net` |
| Flat-LV source slack voltage | 1.00 pu | `build_lv_net` |
| Power-factor clamp | [0.5, 1.0] | `_pq_kw` |
| Shunt capacitance `c_nf_per_km` | 0 | `_add_line` |
| Sizing max iterations | 8 | `size_conductors` |
| Non-convergence bail streak | 2 | `size_conductors` |
| Hours per year | 8760 | loss energy |

> **Modelling limitations** (see README): balanced 3-phase only; single fixed source; heuristic
> road-pole + MST topology; leaf-branch-level (not per-customer) stand-alone decisions; heuristic MV/LV
> zoning and transformer placement; no explicit step-up transformer; illustrative cost values.
