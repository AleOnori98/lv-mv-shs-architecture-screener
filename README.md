# LV/MV + Stand-alone Mini-Grid MVP

A lightweight, interactive **Streamlit** tool that compares four rural electrification
architectures for a settlement and recommends one by **least-feasible net present cost (NPC)**:

- **A — Full LV**: every customer on one low-voltage network.
- **B — LV + Stand-alone**: terminal **leaf branches** cheaper to serve off-grid become solar
  stand-alone systems; the LV grid is rebuilt and resized on the remainder.
- **C — MV/LV service zones**: the settlement is partitioned into capacity-limited transformer
  service zones (one DT each), connected by an MV trunk — the realistic answer for large villages
  where a single LV network would collapse.
- **D — MV/LV + Stand-alone**: its **own** per-zone leaf-branch stand-alone screen is applied to
  Case C's zones, then the remaining grid-served demand is re-partitioned into MV/LV zones.

It reuses a road-pole + **MST** topology engine and validates every network with **pandapower**
(balanced three-phase, present and horizon demand).

> **This MVP is a sequential diagnostic workflow, not a full co-optimization model.**

📖 **Docs:** [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — plain-language, step-by-step walkthrough ·
[`docs/MODEL_REFERENCE.md`](docs/MODEL_REFERENCE.md) — exact parameters, variables & equations.

---

## How it works

```
CSV / GeoJSON ─▶ load_inputs ─▶ customers_to_engine_gdf
        │
        ▼
  lv_topology (road-pole + MST engine)
        │
        ▼
  pandapower_builder + conductor_sizing + validation ─▶ Case A (Full LV)
        │
        ├─▶ terminal-subtree screen ─▶ standalone ─▶ rebuild ─▶ Case B (LV + Stand-alone)
        └─▶ mvlv_service_zones ─▶ transformers + MV trunk ─▶ Case C (MV/LV)
             └─▶ per-zone terminal-subtree screen ─▶ rebuild ─▶ Case D (MV/LV + Stand-alone)
        │
        ▼
  cost_model + npc ─▶ mvp_runner compares ─▶ tables · maps · downloads
```

1. Generate a Full-LV topology connecting every customer.
2. Validate in pandapower at the **diversified design peak** (loads scaled by a **coincidence factor**,
   default 0.6) with the cheapest LV conductor, then **iteratively upgrade** conductors (by `rank`) on
   critical lines until voltage/loading limits pass.
3. Generate SHS candidates as **terminal subtrees** (leaf branches) of the LV tree — every maximal
   branch hanging off a fork. Engineering thresholds (distance, cost, voltage) are reported as
   diagnostics only; they no longer gate the decision.
4. For each eligible branch, compute a **stand-alone NPC** (cheapest qualifying solar package) and
   select it **iff** `SHS_NPC < avoided grid cost` (avoided LV branch, service-drop, pole, connection
   and — for the LV screen — loss costs). Case D runs the same screen **per MV/LV zone**.
5. Build the four candidate architectures and validate each at present and horizon demand, plus a
   **coincident-peak stress** snapshot (every customer at their peak at once) reported for robustness.
6. Compare by NPC — which now includes the **mini-grid generation cost** (a levelised LCOE on grid-served
   energy, so grid and stand-alone are costed on the same footing) — and report a recommendation, a
   per-customer strategy table, a decision log, a **tier-served / future-expansion** view, coincident-peak
   stress indicators, and a **parallel-coordinates** trade-off plot of the feasible solutions.

---

## Project layout

```
app.py                         Landing page
pages/1_MVP_Topology_Planning.py   Full workflow UI
core/
  io/        input schemas, loaders, example dataset
  topology/  LV topology wrapper, graph utils, terminal-subtree + branch detection, MV/LV zones
  electrical/ pandapower builder, conductor sizing, validation
  economics/ cost model, stand-alone, NPC
  workflow/  mvp_runner (orchestrator), result models, exports
  viz/       folium maps, charts (NPC + SHS screening), table styling
  distribution_{io,algos,service}.py   Reused road-pole + MST topology engine
config/settings.py             Engine constants
docs/                          MODEL_REFERENCE.md (equations) · WORKFLOW.md (walkthrough)
examples/mvp/                  Runnable example dataset (60 customers)
scripts/                       Example generator + smoke check
tests/                         pytest suite
```

The topology engine entry point reused by the MVP is
`core.distribution_service._run_low_voltage_core`; everything else under `core/` is new.

---

## Install & run

```bash
conda env create -f environment.yml      # creates the 'lvmv-mvp' env
conda activate lvmv-mvp
streamlit run app.py                      # open "MVP Topology Planning" in the sidebar
```

Pick a dataset, then **Load example dataset → Run simplified LV/MV + stand-alone MVP**.

### Example datasets

Any folder under `examples/` containing the required CSVs is auto-discovered and listed in the
page's dataset selector.

| Dataset | Size | What it shows |
|---------|------|---------------|
| **Synthetic demo** (`examples/mvp/`) | 60 customers, ~15 s | Dense core + remote leaf branches + institutional cluster → **B (LV + Stand-alone)** wins (≈13 customers moved off-grid) |
| **NGA 1 compact village** (`examples/nga_1/`) | 128 customers, ~15 s | Real Nigerian customer coordinates + OSM roads, with aggregate village demand distributed across households. Compact layout — **A** and **B** are essentially tied; **B (LV + Stand-alone)** edges A by moving one remote customer off-grid |
| **Mafa case study, NGA** (`examples/mafa case study/`) | 240 customers, ~1.5 min | Real field-survey coordinates + customer types + OSM roads. Flat LV is infeasible; **D (MV/LV + Stand-alone)** wins, moving ≈16 leaf-branch customers off-grid and serving the rest with transformer zones |
| **Realistic SSA village** (`examples/complete/`) | 521 customers, ~2.5 min | Real coordinates, customer types and OSM roads. Flat LV collapses (infeasible); among feasible candidates **D (MV/LV + Stand-alone, ~15 transformer zones)** narrowly edges **C** |

Regenerate them deterministically:

```bash
python scripts/generate_mvp_examples.py        # synthetic demo
python scripts/build_mafa_example.py           # Mafa village (from consumers_coordinates_nga2.xlsx + category_profiles.csv + roads)
python scripts/build_realistic_example.py      # realistic SSA village (from complete/users.xlsx + roads.gpkg)
```

The larger villages are heavy because each candidate is validated at both present and horizon
demand. The raw `examples/nga_{2,3}/` folders (real Nigerian villages, coordinates + roads only)
are kept as GIS references for future use.

Smoke-test any dataset end-to-end: `python scripts/smoke_dataset.py mafa` (or `mvp` / `complete`).

Run the tests:

```bash
python -m pytest tests/ -q
```

---

## Input files

Uploaded in the page (or use the bundled example in `examples/mvp/`). Column schemas and the
applied defaults are defined in [`core/io/input_schemas.py`](core/io/input_schemas.py).

| File | Purpose |
|------|---------|
| `customers.csv` | `customer_id, latitude, longitude, customer_type, peak_kw_now, annual_kwh_now, power_factor, growth_rate, demand_tier, priority_weight, must_connect_flag, can_be_standalone_flag` |
| `source.csv` | single injection point + LV/MV voltages |
| `lv_conductors.csv` / `mv_conductors.csv` | catalogue with `r/x_ohm_per_km, max_i_ka, cost_per_m, rank` (rank 1 = cheapest) |
| `transformers.csv` | ratings + impedances + cost + `rank` |
| `standalone_packages.csv` | solar packages by `service_tier`, peak, energy, capex, lifetime |
| `cost_parameters.csv` / `.yaml` | discount rate, horizon, unit costs, **generation LCOE**, **coincidence factor**, thresholds, limits |
| `roads.geojson` / `.gpkg` | optional; seeds pole placement |

---

## Outputs

In-app: recommended architecture and savings, a candidate comparison table, an **NPC-by-component
bar chart** (feasible candidates, now including a **Generation (LCOE)** component), per-candidate
detail (cost-share donut + breakdown metrics, **headroom bars** for voltage/loading at both the design
and the **coincident peak**, violation banners), the **SHS screening** panel (per leaf-branch economics
for B and D), a **coincident-peak stress** comparison, a **tier-served & future-expansion** panel (grid
connections stay expandable, SHS capped at their package tier), a **parallel-coordinates** trade-off plot
of the feasible solutions (total NPC · total tier capacity · coincident-peak min voltage), a
customer-strategy table, a decision log, four candidate maps (customers colored grid / stand-alone /
**unserved**, with service-zone areas and a selectable **coincident-peak** map scenario), and downloads
(`comparison_table.csv`, `customer_strategy.csv`, `decision_log.csv`, `cost_breakdown.csv`,
`*_edges.geojson`, `transformers.geojson`, `standalone_customers.geojson`,
`pandapower_validation_results.csv`, and `all_outputs.zip`).

**Interpreting it:** the recommended candidate is the **feasible** one with the lowest
`npc_total`; if none is feasible, the least-infeasible candidate and its active bottleneck are
reported. The decision log records every leaf-branch / customer decision with the driving metric.

---

## Assumptions & limitations

- Balanced three-phase approximation only; no unbalanced/phase allocation.
- Single fixed source/injection point.
- Topology via road-pole + MST heuristic (Euclidean distances; roads seed pole locations).
- Stand-alone decisions are leaf-branch-level (a whole terminal subtree is kept or cut), not exact
  per-customer optimization.
- MV/LV uses heuristic capacity-based zoning (greedy farthest-first, bounded by zone radius and
  transformer rating) with transformer placement at the zone centroid (road-snapped) — not a
  facility-location optimizer. A small fraction of customers may remain unserved inside a zone
  (reported as `service_tier_violations`).
- The MV source/slack is set at a realistic high sending voltage (~1.04 pu) to offset MV +
  transformer drop; no explicit step-up transformer is modelled.
- Demand diversity is a **single flat coincidence factor** for the whole system (no size-dependent
  ADMD); the network is designed at that diversified peak and stress-tested at the coincident peak.
- Demand growth is stress-tested through horizon power-flow validation, not optimized dynamically; a
  single peak-power snapshot is used (no hourly/time-series dispatch).
- Mini-grid generation is costed with **one flat LCOE** applied to grid-served energy — a proxy for a
  dispatch/generation model, not an hourly production simulation.
- Cost values are illustrative unless replaced with local data.

---

## Next steps

The MVP is deliberately a **single-snapshot, deterministic diagnostic**. The most valuable extensions
turn each of its current simplifications into a richer evaluation axis. Roughly in priority order:

**1. Robustness under demand uncertainty (a new evaluation criterion).**
Today every candidate is judged on one demand realisation. The natural next step is to run each
architecture across **multiple demand scenarios** (e.g. low / central / high growth, different tier
mixes, adoption timing) and score not just NPC but **robustness** — how stable a solution's feasibility
and cost stay across scenarios (worst-case NPC, regret, probability-of-feasibility). Robustness would
then join total NPC, tier capacity and stability as its own axis in the parallel-coordinates plot, so
the recommendation can favour solutions that are good *and* resilient to what the future actually does.

**2. Hourly load profiles per customer → time-series stress testing.**
The load model is currently a peak snapshot scaled by a flat coincidence factor. Attaching an **hourly
(8760 or representative-day) profile per customer/type** — several datasets already ship
`category_profiles.csv` — would let the power flow be evaluated not only at the coincident peak but at
**other relevant hours** (evening lighting peak, midday productive/solar-aligned load, minimum load for
over-voltage checks). This replaces the flat coincidence factor with **emergent diversity** from the
profiles and makes the stress test a true multi-hour analysis rather than a single worst case.

**3. A generation/dispatch model behind the LCOE.**
The flat `generation_lcoe_per_kwh` is a proxy. With hourly profiles in place it can be replaced by a
lightweight **generation-sizing / dispatch** step (PV + battery + backup, à la a MicroGridsPy surrogate)
whose LCOE **scales with served demand and load shape** — capturing economies of scale and the fact
that a spikier, less-diversified load costs more to generate.

**Further ideas:** size-dependent **ADMD / diversity curve** (replace the single coincidence factor with
one that depends on the number of aggregated customers); **reliability metrics** (LOLP, expected unserved
energy) once profiles exist; **unbalanced / phase-aware** power flow; **multi-source** siting instead of a
single fixed injection point; **per-customer** (not leaf-branch) stand-alone decisions moving toward true
co-optimization; and **sensitivity / tornado** analysis on the illustrative cost parameters.

---

## Contact

**Alessandro Onori** — alessandro.onori@polimi.it
Based on original topology work by **Edoardo Silvestri**.
Technical advisors: Riccardo Mereu, Emanuela Colombo (Politecnico di Milano).

## License

European Union Public Licence (EUPL v1.1).
