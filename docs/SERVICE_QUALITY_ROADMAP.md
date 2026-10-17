# SHS costs, service quality and policy outputs: next steps

**Why this document.** The convergence experiment
([`CONVERGENCE_EXPERIMENT.md`](CONVERGENCE_EXPERIMENT.md)) showed that the SHS-vs-grid balance on
Mafa is not set by the generation curve. Three other things set it:

- the SHS package costs;
- which package each household qualifies for;
- the fact that the model compares an SHS and a grid connection on cost alone.

This document records what that implies and fixes the order of work. **The bottom-up SHS cost
estimate waits for the RAMP integration.** Steps 1–4 below don't need it and can go first.

---

## 1. SHS costs are probably too low, and the answer depends on them

### Evidence

The per-branch decomposition from the experiment (Mafa, 0.25 €/kWh, flat generation credit) was
re-scored with every SHS NPC multiplied by a factor:

| SHS cost × | Screen B: branches where SHS wins | Screen D: branches where SHS wins |
|---|---|---|
| 1.0 (catalogue) | 37 / 37 | 42 / 42 |
| 1.5 | 37 / 37 | 42 / 42 |
| 2.0 | 31 / 37 | 28 / 42 |
| 2.5 | 12 / 37 | 6 / 42 |
| 3.0 | 6 / 37 | 4 / 42 |

Each branch is scored on its own (`SHS_NPC × m < network avoided + generation credit`), so the
nesting and the greedy selection are ignored. The table therefore shows how sensitive the screen
is; it is not a rerun of the pipeline. Reproduce from the saved results:

```python
import json
d = json.load(open("docs/figures/convergence/convergence_results.json"))["decomposition"]
for m in (1, 1.5, 2, 2.5, 3):
    print(m, {s: sum(m * r["shs_npc"] < r["network_avoided"] + r["generation_credit_flat"]
                     for r in d[s]) for s in "BD"})
```

**The result holds up to 1.5× and collapses between 2× and 2.5×.** No Mafa conclusion about how
many households go off-grid should be cited until the SHS costs are calibrated.

### Why the catalogue looks low

`SHS_T2` claims 800 kWh/yr and 0.45 kW for 520 € capex. That is about 2.2 kWh a day. In the
Sahel this needs roughly 0.6–0.7 kWp of PV and, because most household use is in the evening,
about 2 kWh of usable battery, plus an inverter, installation and a distributor margin. A rough
engineering estimate is **1,000–2,000 € installed**. This is an estimate to check against
published data (GOGLA price data, ESMAP mini-grid reports, IRENA), not a sourced figure.

The levelised result points the same way. The model prices a typical household's SHS at about
**0.22 €/kWh**, below the mini-grid's 0.25 €/kWh, and that 0.25 covers generation only. Published
comparisons usually find the opposite: SHS is dearer per kWh and cheaper per connection, which is
why it wins for small, scattered loads. An SHS that wins on cost per kWh most likely reflects the
input data, not a finding.

The package choice adds its own sensitivity. The 2% `standalone_sizing_tolerance` is what lets the
median household (sized for year-15 demand, ≈ 0.45 kW) fit `SHS_T2` instead of `SHS_T3`.

### Plan: bottom-up SHS cost (after RAMP)

Price the SHS with the same tools as the mini-grid, so that both options deliver the same service
by construction:

1. For each RAMP household archetype, run **MicroGridsPy on a single-household load**:
   - the same PVGIS typical year;
   - the same unmet-demand (LLP) limit as the mini-grid;
   - small-scale component prices (PV, LFP battery, charge controller/inverter, installation,
     distribution margin, and the cost of PAYGo financing where relevant).
2. The result is a cost function per archetype, `SHS_NPC(archetype, LLP target)`. It replaces the
   fixed `standalone_packages.csv` catalogue and its eligibility rule. Keep the catalogue as an
   input format for users who have real quotes.
3. Size for the horizon the same way the grid is sized: year-T demand, or a staged replacement
   when the battery is replaced.

Seam: `choose_package_for_customer` and `cluster_standalone` in `core/economics/standalone.py` are
the only places that turn a customer into an SHS cost. A cost-model object like the generation seam
(`core/economics/generation.py`) can replace them without touching the screen.

**Interim (before RAMP):** report every SHS result as a range by rerunning at SHS cost ×1, ×2 and
×3. A single `standalone_capex_multiplier` parameter is enough for this.

---

## 2. Service quality: what the model has, what it lacks

### Already in place

- **A capacity floor.** A package qualifies only if `service_tier ≥ demand_tier`, and it is sized
  for year-T demand.
- **A tier report.** `_tier_service_table` (`core/workflow/mvp_runner.py`) lists each customer's
  service ceiling, headroom and whether they are capped. It is shown in the UI.
- **Protected loads.** `GRID_PRIORITY_TYPES` (clinic, hospital, school, productive, water pump)
  can never go off-grid.

### Missing

| Gap | Effect |
|---|---|
| **Availability and reliability** | The SHS is assumed to meet 100% of demand: no autonomy days, no harmattan dust, no rainy-season shortfall. The flat LCOE assumes the mini-grid never fails either. Both are optimistic, probably not by the same amount. |
| **Tier labels don't match ESMAP's Multi-Tier Framework (MTF)** | A Mafa "tier 2" household (400 kWh/yr ≈ 1.1 kWh/day, 0.25 kW) is MTF **Tier 3** by capacity (≥ 1 kWh/day, ≥ 200 W). `SHS_T2` (≈ 2.2 kWh/day, 450 W) is also Tier 3. The labels are one tier low at the bottom end, which matters once tiers are reported outside the model. |
| **Growth ceiling and lock-in** | A capped SHS stops demand growth at the package limit, and SHS households are rarely connected later. The choice is static over 15 years, with no value for keeping the grid option open. |
| **Which loads may go off-grid** | Mosques, shops and public-service buildings can currently go to SHS. That is a policy choice the model makes without showing it. |

Check the MTF thresholds against the source framework (Bhatia & Angelou, *Beyond Connections*,
ESMAP 2015) before relabelling. Its availability and reliability attributes matter as much as
capacity.

### How to bring service quality into the decision

Ranked from most to least defensible:

1. **Make the service equal by construction.** Size the SHS and the mini-grid to the same LLP
   target with the same dispatch model (§1 plan). This needs no weights. It is the target design,
   and it depends on RAMP.
2. **Hard floors as scenarios.** For example, "MTF Tier ≥ 3 for everyone", "Tier ≥ 4 for these
   building types", or "no SHS for type X". The normative choice stays visible and is set by the
   user, not the model.
3. **A price on unserved energy (value of lost load).** This is defensible only if both options
   are simulated hour by hour, and the value itself is contested. Use it as a sensitivity, not a
   default.
4. **Avoid an arbitrary €-per-tier-gap penalty.** The answer becomes whatever weight the analyst
   picks. It also double-counts if combined with option 1 or 2.

---

## 3. Policy outputs: report the price of service quality

The model should not build a value judgement into its objective. It should report **what a value
judgement would cost**. Three outputs do that, and none of them needs a penalty weight.

### 3.1 Switching value per branch

For each selected off-grid branch `b` with `n_b` customers, at the converged state:

```
subsidy_b = (avoided_b - SHS_NPC_b) / n_b          [€ per connection, present value]
value_b   = subsidy_b / AF(r, T)                   [€ per household per year]
```

`subsidy_b` is the grant per connection that would make keeping branch `b` on the grid break even.
Sort the branches by `subsidy_b` and accumulate households to get a **supply curve of grid
service**: households kept on the grid against the subsidy per connection. That curve maps onto
results-based-financing grants per connection.

Notes:

- Compute it on the final, disjoint selection. Nested candidates make the per-branch values
  non-additive.
- `avoided_b` includes the generation credit, which depends on who else is on the grid. Report the
  value at the converged state of the coupling loop, not the first round.
- This needs no new modelling: the screen already computes `avoided_b` and `SHS_NPC_b`.

### 3.2 Cost of a tier floor

Rerun the pipeline under floors such as "Tier ≥ 3 for all" and "Tier ≥ 4 for all", where a floor
excludes any SHS package below it. Report the extra NPC in total and per household against the
least-cost plan. A typical headline: "Tier 4 for every household in Mafa costs +X € (+Y %) over the
least-cost plan."

### 3.3 Distribution of service tiers, and where

Aggregate `_tier_service_table` into the number of people at each MTF tier under each
architecture, plus a map. The least-cost plan gives lower service to the outer branches as a rule.
The map makes that equity effect visible.

### Link to the on-grid extension

An unreliable national grid scores low on MTF availability and reliability. Once outages are
modelled ([`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md) §5.2), the MTF becomes the common measure
across SHS, mini-grid and the national grid. Each option is then described by its cost and the tier
it actually delivers.

---

## 4. Order of work

| # | Step | Needs RAMP? | Main files |
|---|---|---|---|
| 1 | Map `demand_tier` and `service_tier` to MTF capacity thresholds; flag mismatches on input | No | `core/io/input_schemas.py`, example `customers.csv` and `standalone_packages.csv` |
| 2 | `standalone_capex_multiplier`; report results as ×1 / ×2 / ×3 ranges | No | `core/economics/standalone.py`, `core/io/input_schemas.py` |
| 3 | Switching-value output and supply curve (§3.1) | No | `core/workflow/mvp_runner.py`, `result_models.py`, UI page |
| 4 | Tier-floor scenarios and the tier distribution (§3.2, §3.3); make the off-grid exclusion list a parameter | No | `core/economics/standalone.py`, `mvp_runner.py` |
| 5 | **Bottom-up SHS cost** per archetype from single-household MicroGridsPy runs (§1 plan) | **Yes** | new SHS cost seam beside `core/economics/generation.py` |
| 6 | Equal-LLP comparison of SHS and mini-grid (§2, option 1) | Yes | SHS seam and generation seam |
| 7 | Tier of the unreliable grid in the on-grid option | Yes, plus outage data | on-grid work (§5.2 of the review) |

Steps 1–4 make the current results honest and give the policy outputs a working form. Step 5 is the
one that decides the Mafa answer.

## Caveats for policy use

- **The cost data is illustrative**, and the result flips within the plausible SHS cost range.
  Publish ranges and thresholds with sourced costs, never a point estimate.
- **NPC is the society-wide view.** Households see tariffs and PAYGo instalments, and affordability
  is itself an MTF attribute. The two views can rank the options differently.
- **The choice is static.** The 15-year horizon has no option value for a later grid connection.
  MicroGridsPy's multi-year mode is the natural place to add staged decisions.
- **Positioning.** Least-cost electrification tools such as OnSSET take a target tier as an input.
  Building-level resolution, power-flow checks and an explicit price of the tier would add to that,
  but only once the SHS costs are calibrated.
