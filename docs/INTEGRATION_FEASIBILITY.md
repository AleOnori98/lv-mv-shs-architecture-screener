# Integration Feasibility — RAMP + PVGIS + MicroGridsPy + LV/MV/SHS Screener

**Date:** 2026-10-19 · **Scope:** backend only (no GUI), YAML/CSV I/O, typical-year, isolated mode.

A critical assessment of whether the four existing pipelines can be re-shaped into modular,
reusable Python packages and chained into one automated scarce-data planning workflow — and what
it would actually take.

Everything below marked **verified** was executed in a container during this assessment, not
inferred from reading code. Everything marked **unverified** is flagged as such.

---

## 0. Verdict up front

**The packaging and plumbing are feasible and bounded. The iterative loop is the real risk, and
it is a methodological risk, not an engineering one.**

Three findings drive that:

1. **One environment holds all four engines.** *Verified*: Python 3.11 + numpy 1.26.4 +
   pandas 2.1.4 + pandapower 3.1.2 + geopandas 1.1.4 + rampdemand 0.5.0 + microgridspy 0.4.0[highs]
   install together and all run. MicroGridsPy solved its bundled `demo_typical_year` to optimality
   with HiGHS in ~13 s wall (8760 h, 1 scenario, 46,439 simplex iterations). RAMP generated profiles
   in the same interpreter.

2. **The screener's core is already library-shaped.** *Verified*: `core/` contains zero Streamlit
   imports; `run_simplified_mvp()` is a pure function (DataFrames + dicts in, dataclasses out);
   43/43 non-viz tests pass on a modern stack (Python 3.13, numpy 2.4, pandas 2.3). The
   "re-shape into a package" job here is packaging, not rewriting.

3. **The coupling variable already exists on both sides.** MicroGridsPy emits
   `LCOE (Expected, delivered) = reported_total_annual_cost / served_energy_kwh`; the screener
   already consumes a scalar `generation_lcoe_per_kwh` and already credits avoided generation in the
   SHS screen (`mvp_runner._avoided_grid_cost`). The loop closes on a single number that both ends
   speak today.

Against that: **RAMP is the weak link on every axis** (reproducibility, API shape, dependency
ceiling, taxonomy coverage), the screener's **cost model is structurally biased toward the grid**,
and the proposed iteration is a **positive-feedback fixed point with no convergence guarantee**.
Those three must be addressed before the loop produces trustworthy numbers — not after.

Realistic effort for a working headless backend: **6–10 weeks** of focused work, roughly 70%
packaging/schema/wrapper and 30% new modelling.

---

## 1. What is actually there

Assessed from the repository in the working directory plus the three upstream repositories
(`AleOnori98/RAMP-Streamlit`, `AleOnori98/PVGIS-Streamlit-App`, `MicroGridsPy/MicroGridsPy`), which
were cloned for this review. The zip folder described in the task was not present in this
container; the GitHub repositories were used instead and are assumed equivalent.
`OffGridPlanner` was not available and is not assessed.

### 1.1 LV/MV/SHS screener (this repository) — 9,555 LOC

| Aspect | State |
|---|---|
| Streamlit coupling | **None in `core/`** (verified by grep). UI lives only in `app.py` + `pages/`. |
| Entry point | `core/workflow/mvp_runner.run_simplified_mvp()` — pure function, `progress` callback optional. |
| Tests | 43 pass, 2 modules skipped for missing `folium`/`altair` (viz-only). |
| Portability | Runs unchanged on Python 3.13 / numpy 2.4 / pandas 2.3 / pandapower 3.1. |
| Installability | **No `pyproject.toml`, no `setup.py`, no `.gitignore`.** Import works only from repo root. |
| Legacy core | `core/distribution_{algos,service,io}.py` = 2,892 LOC (30% of the codebase) at package root. |
| Public API hygiene | `lv_topology.py` imports `_run_low_voltage_core` — a **private** name used as the public engine entry point. |
| Repo hygiene | **21 `*.py.tmp.*` editor-crash files and 59 `__pycache__` files are tracked in git.** |
| Env pinning | `environment.yml` has **zero version pins**. |

The documentation (`docs/WORKFLOW.md`, `MODEL_REFERENCE.md`, `METHODOLOGY_ENGINES.md`) is unusually
good and honest about its own simplifications — it is a genuine asset and should survive the
repackaging intact.

#### The cost model is asymmetric, and the asymmetry favours the grid

`core/economics/cost_model.CostBreakdown`:

```
total_npc = grid_capex + standalone_npc + loss_pv_cost + generation_npc
```

- **SHS** gets a full lifecycle NPC — `npc.standalone_npc()` = capex + PV(O&M) + PV(replacements) − PV(salvage).
- **Grid assets** get **capex only**. No O&M on lines, poles or transformers. No transformer
  replacement. No salvage credit. Only losses and a flat generation LCOE are added.

So the single number that picks the recommended architecture systematically understates the cost of
grid-served customers, and the understatement grows with the planning horizon. This is a real
methodological defect today, independent of any integration work. It becomes far more serious once
the loop makes the grid/SHS partition endogenous: the iteration would converge onto a fixed point
that is biased by construction.

**Fix this before building the loop, not after.**

#### The hourly dimension is discarded before the core ever sees it

`examples/mafa case study/category_profiles.csv` contains real 8760-hour profiles for 12 consumer
categories. `scripts/build_mafa_example.py` collapses them to `(peak_kW, annual_kWh)` per customer
*offline*, and the core workflow only ever sees those two scalars. `examples/complete/category_profiles.csv`
is a placeholder column of 1s.

Adding time series is therefore a **new capability**, not a plumbing exercise. The hook is clean
though: `pandapower_builder._pq_kw()` is the single place where a customer's power is computed.

### 1.2 PVGIS pipeline — 485 LOC core, the easiest win

Core is Streamlit-free and small. The useful surface is already the right shape:

- `pvgis_client.get_tmy_hourly(lat, lon) -> DataFrame` (`G(h)`, `Gd(h)`, `T2m`, `WS10m`, `WD10m`)
- `solar_model.pv_from_tmy(...) -> DataFrame` indexed 1..8760 with **`pv_kwh_per_kwp`**
- `wind_model.wind_from_tmy(...)` + a turbine power-curve library

`pv_kwh_per_kwp` is *exactly* the per-unit-nominal-capacity series MicroGridsPy's
`resource_availability.csv` wants. This stage is close to drop-in.

Weaknesses: no caching (every run hits the API), no retry/backoff, a bare 30 s timeout, and the
endpoint URL hardcoded in `config/settings.py`. PVGIS versions its API and has broken callers
across minor versions before. For a pipeline whose whole point is scarce data, an on-disk cache and
a pinned API version are not optional.

*Note:* `re.jrc.ec.europa.eu` was unreachable from this container — the environment's network policy
denied the host — so the live API call could not be exercised. The code path was read, not run.

### 1.3 RAMP + SSA archetypes — the weak link, on four separate axes

**(a) Not a library.** `ramp_core.run_single_archetype(excel_path, num_days)` writes CSVs into a
global `PathManager.outputs_dir` as a side effect and returns *file paths*. `run_multi_archetype()`
takes **no arguments at all** — it reads `inputs/year_structure.json` and
`inputs/archetype_configs.json` from fixed locations. Both are unusable as functions in a pipeline
without a rewrite of the call surface.

**(b) Not reproducible — but fixable.** `ramp_core.py` calls `default_rng()` with no seed at lines
155 and 304. Worse, RAMP itself exposes no seed parameter (`UseCase.generate_daily_load_profiles`
has none) and uses **both** the stdlib `random` module (`random.gauss`, `random.normalvariate` in
`ramp/core/core.py`, for peak-time sampling) and numpy.

*Verified empirically*: seeding **both** `random.seed(s)` and `np.random.seed(s)` before the call
makes two RAMP runs bit-identical; without seeding they differ. So the user's "reproducible
stochastic assumptions" requirement **is achievable**, but only via process-global seeding — which
means RAMP runs cannot be safely parallelised across threads with independent seeds. Use processes,
or accept serial.

**(c) Dependency ceiling.** *Verified*: `rampdemand` 0.5.0 is the latest release on PyPI (no newer
fix exists) and it fails on:
- **numpy ≥ 2.0** — `TypeError: only 0-dimensional arrays can be converted to Python scalars`
- **pandas ≥ 2.2** — `ValueError: invalid unit abbreviation: T` (the `"T"` minute alias was removed)

and there is no pandas < 2.2 wheel for Python 3.13. RAMP alone sets the ceiling for the entire
stack at **Python ≤ 3.12, numpy < 2, pandas < 2.2**. Everything else in the stack is happy well
above that. This is the single strongest argument for keeping RAMP in its own distribution
(see §4).

**(d) Taxonomy mismatch — the biggest *data model* problem in the whole plan.**

The bundled archetype library (`config/archetypes_library/`) covers:

```
{AY,AS,NC,OM}_{F1..F5}_Tier-{1..5}.xlsx   households: cooling period × lat zone × wealth tier
HOSPITAL_Tier-{1..5}.xlsx
SCHOOL.xlsx
```

The screener's `CUSTOMER_TYPES` are:

```
household, shop, productive, school, clinic, hospital, mosque, public_service, water_pump, other
```

So **shop, productive, water_pump, mosque, public_service and clinic have no archetype at all** —
and those are precisely the daytime loads that determine whether PV is cheap or expensive for this
village. Household / school / hospital alone give you an evening-peaked profile and a systematically
pessimistic PV case.

`archetypes_core.determine_zone()` also hard-fails outside −30° ≤ lat ≤ 20°, and maps zone from
latitude alone with no actual climate data.

*Unverified*: zenodo.org was unreachable from this container (network policy), so I could not check
whether the published Zenodo record (22832973) ships a broader taxonomy than the copy bundled in the
repo. **Check this first** — it may close part of the gap for free.

### 1.4 MicroGridsPy — the model citizen, and the template to copy

`src/` layout, `pyproject.toml` with hatch-vcs versioning, `pip install microgridspy[highs]`,
a documented `api.py` (`solve`, `solve_example`, `load_results`, `export_results`, `create_project`,
`set_workspace`, `validate_project`, `list_projects`), a CLI entry point, bundled runnable examples,
and inputs as YAML (`renewables.yaml`, `battery.yaml`, `generator.yaml`) + CSV
(`load_demand.csv`, `resource_availability.csv`) + `formulation.json`.

That is exactly the structure the other three should be rebuilt into. **Do not invent a new
convention — copy this one.**

Project input contract (typical year, off-grid):
- `load_demand.csv` — 8760 rows, `;`-delimited, units **kWh per hour**, one column per scenario
- `resource_availability.csv` — 8760 rows, per-unit-nominal production per resource
- `formulation.json` — `system_type: off_grid`, `on_grid: false` (weak-connection is already
  supported here, which de-risks that future requirement)

Output contract: `results.kpis` (a wide DataFrame: `served_energy_kwh`, `reported_total_annual_cost`,
capacities, emissions, …) and a summary table carrying **LCOE (Expected, delivered)**.

---

## 2. The iterative loop — where the scepticism belongs

The proposed coupling is a fixed-point iteration on one scalar:

```
LCOE_k ─▶ SHS screen ─▶ grid-served set S_k ─▶ aggregate demand D_k ─▶ MGPy ─▶ LCOE_{k+1}
```

### 2.1 The feedback is positive. This is the mini-grid death spiral.

Raising LCOE pushes marginal leaf branches off-grid → removes demand → but mini-grid generation has
strong economies of scale and a largely fixed cost base (power house, inverter, control, minimum
genset and battery granularity) → **LCOE rises again**.

Formally `dLCOE/dD < 0` and `dS/dLCOE < 0` compose into a **positive loop gain**. Where the local
gain exceeds 1, plain Gauss–Seidel iteration diverges to "everyone on SHS". There is no proof of a
unique fixed point; there may be several, and the iteration will land in whichever basin the
initial guess sits in.

**A run that converges to all-SHS is a model artefact to be detected, not a planning result.**

Build the defences in from the first version, not as a later fix:

1. **Damping:** `LCOE_{k+1} = λ·LCOE_MGPy + (1−λ)·LCOE_k`, λ ≈ 0.5.
2. **Hysteresis:** a branch only flips grid↔SHS when it clears the threshold by a margin. This is
   what kills 2-cycles (A↔B oscillation), which are the most likely failure mode.
3. **Hard iteration cap** (~8).
4. **Best-iterate-wins:** return the iterate with the **lowest total system NPC among all visited**,
   not the last one. This makes the loop an honest 1-D search instead of a convergence claim.
5. **Log every iterate** (LCOE, |S_k|, demand, total NPC) so oscillation is visible, not hidden.

Runtime is not the constraint: ~13 s per MGPy solve + ~90–150 s per full 4-case screener run on
240–521 customers → ~8 iterations ≈ 15–20 minutes. Acceptable for a batch backend.

### 2.2 The two models do not share a financial world

MGPy minimises annualised cost using per-technology WACC and its own horizon. The screener
minimises a 15-year NPC at a single `discount_rate`. Feeding MGPy's LCOE into

```
generation_npc = lcoe × annual_kwh × annuity_factor(discount_rate, planning_horizon_years)
```

is only coherent if `discount_rate` and `planning_horizon_years` match MGPy's financial parameters.
Today they are two independent parameter sets that can silently disagree, and nothing detects it.

**One shared financial object, consumed by both, is a prerequisite — not a refinement.**

### 2.3 Demand diversity will be double-counted unless a convention is fixed

The screener sizes the network at `Σ peak_i × coincidence_factor` (default 0.6). MGPy receives a
community-**aggregate** hourly profile whose peak already embeds all diversity.

- If RAMP per-type profiles are divided by N to get per-customer peaks, that per-customer "peak" is
  *already a diversified average*. Multiplying by 0.6 again **under-sizes the LV network**.
- If RAMP is run per customer (`num_users=1` — `User.generate_single_load_profile` exists, so this
  is supported), you get true individual peaks and the coincidence factor should be **dropped** in
  favour of emergent diversity from the profiles.

Pick one convention, write it into the schema, and enforce it. Benchmark per-customer RAMP for 500
customers early — if it runs in minutes, do it properly.

### 2.4 Call the loop what it is

The only thing LCOE changes is the grid/SHS partition. MV/LV zoning, conductor sizing and pole
placement are deterministic functions of the customer set. So this is a **1-D fixed point on a
single scalar**, not co-optimisation and not really "iterative soft-coupling" in the strong sense.
That is a perfectly defensible simplification — the screener's own README is admirably honest about
being "a sequential diagnostic workflow, not a full co-optimization model", and that honesty should
carry into any paper describing the coupled version.

---

## 3. Proposed architecture

Four installable distributions plus one orchestrator — **not a monorepo blob**.

```
ramp-archetypes   RAMP + SSA archetype library. Seeded, in-memory, no file side effects.
resource-pvgis    PVGIS TMY fetch + PV/wind conversion. Cached, retried, version-pinned.
gridscreen        This repo, repackaged: topology + power flow + SHS screen.
microgridspy      Unchanged, from PyPI.
ruralplan         Orchestrator: scenario YAML, the fixed-point loop, run dir, provenance.
```

**Why separate distributions rather than one package:** the RAMP pin ceiling (§1.3c). Isolated in
its own distribution, RAMP's `numpy<2, pandas<2.2` constraint can later be escaped by patching,
vendoring or running it out-of-process, without freezing the other three at pandas 2.1 forever.
In one package, the whole stack inherits the ceiling permanently.

**Stage boundaries are files.** Every stage must be runnable alone, reading and writing CSV/YAML.
This is what delivers the "always allow real manual inputs, optionally estimate" requirement — it
falls out of the design rather than needing special-casing. A user with measured load data simply
supplies `demand/profiles_by_type.csv` and never invokes `ramp-archetypes`.

Canonical interchange:

```
scenario.yaml                      single source of truth: discount rate, horizon, currency, seed
demand/profiles_by_type.csv        8760 × consumer type, kW
demand/profiles_by_customer.csv    optional, 8760 × customer (per-customer RAMP path)
demand/customers.csv               screener schema + profile_id
resource/pv_per_kwp.csv            8760, per unit nominal
resource/wind_per_kw.csv           8760, per unit nominal
catalogue/{lv,mv}_conductors.csv, transformers.csv, standalone_packages.csv
provenance.json                    package versions, input hashes, seed, PVGIS fetch timestamp
```

---

## 4. Intervention steps, in order

Ordering matters: phases 0 and 2 are prerequisites that are cheap now and expensive later.

### Phase 0 — hygiene and packaging (≈1 week)

| # | Step |
|---|---|
| 0.1 | Delete the 21 tracked `*.py.tmp.*` files and 59 tracked `__pycache__` files; add a `.gitignore`. |
| 0.2 | Add `pyproject.toml`; move `core/` → `src/gridscreen/`; move `distribution_*.py` into `gridscreen/topology/engine/`; rename `_run_low_voltage_core` → `run_lv_core` and make it genuinely public. |
| 0.3 | Pin the environment at the **verified** joint point: Python 3.11, numpy 1.26.4, pandas 2.1.4, pandapower 3.1.2, geopandas 1.1.4. Add a lockfile. |
| 0.4 | Move `folium`/`altair` behind a `[viz]` extra so `pytest` passes on a core-only install. |
| 0.5 | Add a headless CLI: `gridscreen run --inputs <dir> --out <dir>`, doing what the Streamlit page does today. **This alone makes the screener usable in a pipeline.** |

### Phase 1 — make the inputs real (≈2 weeks)

| # | Step |
|---|---|
| 1.1 | `resource-pvgis` package: `get_tmy(lat, lon, cache_dir)` with on-disk cache, retry/backoff, pinned API version; emit `pv_per_kwp.csv` / `wind_per_kw.csv` directly in MGPy's layout. |
| 1.2 | `ramp-archetypes` package: `simulate(spec, seed) -> DataFrame`. No global paths, no file side effects. Seeds **both** `random` and `np.random`. Replace Excel-only input with a YAML/CSV appliance schema; keep an Excel reader for legacy files. |
| 1.3 | **Decide and document the diversity convention** (§2.3). Benchmark per-customer RAMP at 500 customers. |
| 1.4 | Check the Zenodo record first, then close the taxonomy gap (§1.3d): extend the archetype library or define explicit, documented fallbacks for shop / productive / water_pump / mosque / public_service / clinic. |

### Phase 2 — fix the economics *before* closing the loop (≈1 week)

| # | Step |
|---|---|
| 2.1 | Add grid O&M, replacement and salvage to `CostBreakdown` so grid and SHS are costed symmetrically (§1.1). |
| 2.2 | Make discount rate and horizon a single shared object consumed by both the screener and the MGPy project writer (§2.2). |
| 2.3 | Regression-run the four bundled case studies and record how the recommendation changes. **Expect at least one to flip** — Mafa currently recommends D on a grid-favouring cost model. |

### Phase 3 — time series into the screener (≈2 weeks)

| # | Step |
|---|---|
| 3.1 | Add an optional hourly dimension at `pandapower_builder._pq_kw()`; validate at a handful of selected hours (evening peak, midday PV-coincident, minimum load for over-voltage) instead of one coincidence-factored snapshot. |
| 3.2 | Replace the flat coincidence factor with emergent diversity from the profiles; keep the old path behind a flag so results stay comparable. |

### Phase 4 — the loop (≈2 weeks)

| # | Step |
|---|---|
| 4.1 | `ruralplan` orchestrator: scenario YAML → RAMP + PVGIS → MGPy → screener → LCOE → repeat. |
| 4.2 | Implement damping, hysteresis, iteration cap, best-iterate-wins, full iterate log (§2.1). |
| 4.3 | Run on Mafa (240 customers) and the 521-customer village and **publish the convergence traces**. Whether they converge is the actual deliverable of this phase — a clean divergence result is a finding, not a failure. |

### Phase 5 — later

Weak-connection mode (MGPy already supports `on_grid`, so this is mostly a formulation-flag and a
tariff/reliability input); OSM building ingestion; multi-year formulation; scenario robustness as
already sketched in the screener README's "Next steps".

---

## 5. Things worth pushing back on

- **"Iterative soft-coupling" oversells it.** It is a 1-D fixed point on LCOE (§2.4). Describe it
  accurately in papers; the existing documentation's honesty is a strength worth preserving.
- **Don't start with OpenStreetMap.** Both real case studies already carry surveyed coordinates
  *and* consumer types. OSM gives you building footprints without consumer type or demand — which
  is the hard part, and the part the archetype taxonomy gap (§1.3d) already blocks. It is a
  nice-to-have for new sites, not an input to the core pipeline. Keep it in Phase 5.
- **Resist one big package.** The RAMP pin ceiling makes distribution boundaries load-bearing (§3).
- **Fix the cost asymmetry before the loop.** Tuning a feedback loop around a known structural bias
  wastes the tuning effort and produces a confidently wrong fixed point.

---

## 6. Evidence log

| Claim | How established |
|---|---|
| Joint environment works | Installed py3.11 + numpy 1.26.4 + pandas 2.1.4 + pandapower 3.1.2 + geopandas 1.1.4 + rampdemand 0.5.0 + microgridspy 0.4.0[highs]; all import and run. |
| MGPy solves typical year | `mgp.solve_example("demo_typical_year", solver="highs")` → Optimal, obj 2.267e+05, ~13 s. |
| Screener tests pass | `pytest tests/ --ignore=test_maps --ignore=test_viz_charts` → 43 passed, py3.13/numpy 2.4/pandas 2.3. |
| `core/` is Streamlit-free | `grep -rn streamlit core/ config/` → no matches. |
| RAMP reproducible only with dual seeding | Minimal RAMP `UseCase` run twice: seeded (`random.seed`+`np.random.seed`) → identical; unseeded → different. |
| RAMP breaks on numpy ≥ 2 | `TypeError: only 0-dimensional arrays can be converted to Python scalars` under numpy 2.4.6. |
| RAMP breaks on pandas ≥ 2.2 | `ValueError: invalid unit abbreviation: T` under pandas 3.0.6. |
| rampdemand 0.5.0 is latest | `pip index versions rampdemand` → 0.5.0, 0.4.0, 0.0.0. |
| Repo hygiene | `git ls-files` → 21 `*.tmp.*`, 59 `__pycache__`; no `.gitignore`, no `pyproject.toml`. |
| Archetype taxonomy gap | Directory listing of `config/archetypes_library/` vs `CUSTOMER_TYPES` in `core/io/input_schemas.py`. |
| Cost asymmetry | Read of `core/economics/cost_model.py` vs `core/economics/npc.py`. |

**Not verified** (blocked by the environment's network policy, which denied `zenodo.org` and
`re.jrc.ec.europa.eu`): the Zenodo archetype library's actual taxonomy, and a live PVGIS API call.
Both should be checked on a machine with open outbound access before committing to Phase 1.
