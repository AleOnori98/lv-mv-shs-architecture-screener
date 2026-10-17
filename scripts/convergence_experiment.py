"""Does the generation-cost coupling loop settle? A synthetic-curve experiment on Mafa.

The question
------------
Who gets the grid and who gets a solar home system depends on what generation costs, and
what generation costs depends on how much the plant serves. Coupling the network screen to
a sizing/dispatch model (MicroGridsPy) therefore needs a loop: screen -> served load ->
plant cost -> screen again. This script checks, before any MicroGridsPy integration,
whether that loop converges, cycles, or runs away.

The synthetic generation model
------------------------------
MicroGridsPy is replaced by a curve with economies of scale, in levelised energy E (each
customer's annual kWh discounted over the horizon at its own growth rate, i.e. the same
quantity the flat LCOE prices):

    C(E) = F + k * E**alpha           (C = present value of generation, E > 0)

calibrated so that at the full village C equals the flat-LCOE cost. The two models agree
at full scale and differ only in shape, so any change in the answer comes from the shape:

    fixed_share  F / C(E_full)        share of plant cost that does not scale with energy
    alpha        scale exponent       1 = no scale economies in the variable part

(fixed_share=0, alpha=1) is exactly today's flat LCOE, and is used as a harness check: it
must reproduce the production screen's selection.

The loop
--------
Only the generation term of a branch's avoided cost depends on the state; the network
terms (lines, poles, drops, connections, losses) are properties of the fixed candidate
tree. Each round:

  1. for every candidate branch, generation credit = C(E_ref + e_off) - C(E_ref - e_on),
     where e_on / e_off are the branch's energy currently on / off the grid -- the plant
     with the branch minus the plant without it, valid in both directions;
  2. run the production selection (``_select_standalone_clusters``: greedy, nested,
     laminar) with those credits;
  3. the new off-grid set gives the new served energy E_new;
  4. E_ref <- E_ref + damping * (E_new - E_ref)   (damping = 1: plain substitution).

Stops when the off-grid set repeats from the previous round with E_ref settled
(converged), repeats an older round (cycle), or after ``MAX_ROUNDS``.

The network is NOT rebuilt inside the loop: the candidate branches and their network
costs come from Case A's tree (B screen) and Case C's zone trees (D screen). The realised
Case-D candidate is rebuilt once, on the converged set, to price the outcome.

Run:  python scripts/convergence_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.economics.generation import FlatLcoeGeneration, ServedEnergy  # noqa: E402
from core.economics.npc import growth_annuity_factor  # noqa: E402
from core.economics.standalone import cluster_standalone  # noqa: E402
from core.io.example_data import load_example_inputs  # noqa: E402
from core.io.load_inputs import customers_to_engine_gdf  # noqa: E402
from core.workflow.mvp_runner import (  # noqa: E402
    _avoided_grid_cost,
    _run_mvlv,
    _select_standalone_clusters,
    build_context,
    detect_mvlv_subtree_candidates,
    run_simplified_mvp,
)

OUT = ROOT / "docs" / "figures" / "convergence"
MAX_ROUNDS = 30
ENG = {"max_user_connection_radius_m": 40, "max_users_per_pole": 12,
       "max_pole_span_m": 45, "road_pole_spacing_m": 35}

# Synthetic curve shapes: (fixed_share, alpha). (0, 1) is the flat LCOE. The first pass of
# this experiment used only gentle shapes at the dataset's 0.25 EUR/kWh and found that NO
# decision moved: on Mafa the SHS-vs-grid margins are wide, so the loop was never exercised.
# The sweep therefore also lowers the price LEVEL, down to where branches sit on the edge.
SHAPES = [(0.0, 1.0), (0.15, 0.85), (0.30, 0.70), (0.40, 0.60)]
LCOE_LEVELS = [0.25, 0.20, 0.15, 0.12, 0.10]   # flat-equivalent cost at the full village [EUR/kWh]
DAMPING = [1.0, 0.5]


@dataclass(frozen=True)
class ScaleEconomyGeneration:
    """Synthetic stand-in for a sizing/dispatch model: C(E) = F + k * E**alpha.

    Implements the ``GenerationCostModel`` protocol, so it can be dropped into the same
    seam MicroGridsPy will use. ``E`` is levelised energy -- per-customer annual kWh times
    its own growth annuity factor -- so with F = 0 and alpha = 1 it is the flat LCOE.
    """

    fixed: float
    k: float
    alpha: float

    @classmethod
    def calibrated(cls, fixed_share: float, alpha: float, flat_lcoe: float, e_full: float):
        """Shape (fixed_share, alpha) that costs the same as the flat LCOE at ``e_full``."""
        c_full = flat_lcoe * e_full
        return cls(fixed=fixed_share * c_full,
                   k=(1.0 - fixed_share) * c_full / e_full ** alpha, alpha=alpha)

    def cost(self, e_lev: float) -> float:
        """Present value of the plant serving levelised energy ``e_lev`` (0 if nothing served)."""
        return 0.0 if e_lev <= 0.0 else self.fixed + self.k * e_lev ** self.alpha

    def present_value(self, served: ServedEnergy, *, horizon_years: int, discount_rate: float) -> float:
        e = sum(kwh * growth_annuity_factor(discount_rate, g, horizon_years)
                for kwh, g in served.components)
        return self.cost(e)


def levelised_energy(ctx, customer_ids) -> float:
    """Sum of annual kWh x own growth annuity factor -- the quantity both models price."""
    kwh, growth = ctx["annual_kwh_by_id"], ctx["growth_by_id"]
    return sum(kwh.get(str(c), 0.0) * growth_annuity_factor(ctx["discount"], growth.get(str(c), 0.0), ctx["T"])
               for c in customer_ids)


def run_loop(ctx, clusters, network_avoided: Dict[str, float], screen_ids: List[str],
             model: ScaleEconomyGeneration, damping: float, label: str, cid: str) -> Dict:
    """Iterate the screen against the synthetic plant until the off-grid set settles."""
    energy_of = {str(c): levelised_energy(ctx, [c]) for c in screen_ids}
    e_full = sum(energy_of.values())
    off: frozenset = frozenset()
    e_ref = e_full
    history = [off]
    rounds = []
    outcome = "max_rounds"

    for rnd in range(1, MAX_ROUNDS + 1):
        def avoided(cl, ctx_, grid_ids, e_ref=e_ref, off=off):
            e_on = sum(energy_of.get(str(c), 0.0) for c in cl.customer_ids if c not in off)
            e_off = sum(energy_of.get(str(c), 0.0) for c in cl.customer_ids if c in off)
            gen_credit = model.cost(e_ref + e_off) - model.cost(e_ref - e_on)
            return network_avoided[cl.cluster_id] + gen_credit

        ctx["log"] = []
        sel = _select_standalone_clusters(ctx, clusters, avoided, label, cid,
                                          [c for c in screen_ids if c not in off])
        new_off = frozenset(map(str, sel.customer_ids))
        e_new = e_full - sum(energy_of[c] for c in new_off)
        e_ref = e_ref + damping * (e_new - e_ref)
        rounds.append({
            "round": rnd, "n_off_grid": len(new_off), "served_share": e_new / e_full,
            "avg_cost_per_lev_kwh": model.cost(e_new) / e_new if e_new > 0 else None,
            "n_flipped": len(new_off ^ off),
        })
        settled = abs(e_ref - e_new) <= 1e-6 * e_full
        if new_off == off and settled:
            outcome = "converged"
            break
        if new_off in history[:-1] and settled:
            outcome = f"cycle(period={len(history) - history.index(new_off)})"
            break
        history.append(new_off)
        off = new_off

    return {"outcome": outcome, "rounds": rounds, "final_off": sorted(new_off),
            "n_rounds": len(rounds)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inp = load_example_inputs("mafa")
    cp = inp["cost_parameters"]
    print("Baseline pipeline run (Cases A-D, flat LCOE) ...", flush=True)
    base = run_simplified_mvp(inp["customers"], inp["roads"], inp["source"], inp["lv_conductors"],
                              inp["mv_conductors"], inp["transformers"], inp["standalone_packages"],
                              cp, dict(ENG))
    print(f"  done in {time.time() - t0:.0f}s; recommended {base.recommended}", flush=True)

    # The same context the runner builds, so the screen sees exactly production inputs.
    gdf, crs_meta = customers_to_engine_gdf(inp["customers"])
    T = int(cp["planning_horizon_years"])
    ctx = build_context(
        customers=inp["customers"], params=dict(cp), lv_conductors=inp["lv_conductors"],
        packages=inp["standalone_packages"], horizon_years=T, discount_rate=float(cp["discount_rate"]),
        gdf=gdf, roads=inp["roads"], crs_meta=crs_meta, source=inp["source"], eng=dict(ENG),
        constraints={k: float(cp[k]) for k in ("voltage_min_pu", "voltage_max_pu",
                                                  "max_line_loading_percent",
                                                  "max_transformer_loading_percent")},
        mv_conductors=inp["mv_conductors"], transformers=inp["transformers"],
        lv_kv=float(inp["source"].get("lv_voltage_kv", 0.4)),
        mv_kv=float(inp["source"].get("mv_voltage_kv", 11.0)),
        source_vm_pu=float(cp["source_vm_pu"]),
    )
    flat_lcoe = float(cp["generation_lcoe_per_kwh"])

    screens = {
        "B": {"clusters": base.clusters, "grid": list(map(str, base.cases["A_full_LV"].grid_customer_ids)),
              "losses": True, "label": "LV + Stand-alone", "cid": "B_LV_plus_standalone",
              "production_off": set(map(str, base.cases["B_LV_plus_standalone"].standalone_customer_ids))},
        "D": {"clusters": detect_mvlv_subtree_candidates(base.cases["C_MVLV"], ctx),
              "grid": list(map(str, base.cases["C_MVLV"].grid_customer_ids)),
              "losses": False, "label": "MV/LV + Stand-alone", "cid": "D_MVLV_plus_standalone",
              "production_off": set(map(str, base.cases["D_MVLV_plus_standalone"].standalone_customer_ids))},
    }

    # Network part of each branch's avoided cost: everything except generation.
    no_gen = {**ctx, "generation_model": FlatLcoeGeneration(0.0)}
    for sc in screens.values():
        sc["network_avoided"] = {
            cl.cluster_id: _avoided_grid_cost(cl, no_gen, sc["grid"], include_losses=sc["losses"])
            for cl in sc["clusters"] if cl.standalone_eligible_precheck
        }
        sc["e_full"] = levelised_energy(ctx, sc["grid"])

    # What each SHS decision rests on. For every eligible branch: the SHS package NPC, the
    # network cost it avoids, and the generation credit at the flat price. A branch whose
    # network cost alone exceeds its SHS NPC goes off-grid whatever generation costs; for
    # the rest, `critical_credit_share` is the fraction of the flat generation credit it
    # needs -- the curve has to push the credit below that before the branch flips back.
    flat_ctx = {**ctx, "generation_model": FlatLcoeGeneration(flat_lcoe)}
    for name, sc in screens.items():
        rows = []
        for cl in sc["clusters"]:
            if cl.cluster_id not in sc["network_avoided"]:
                continue
            ctx["log"] = []
            sa = cluster_standalone(cl.customer_ids, ctx["customers"], ctx["packages"], horizon_years=T,
                                    discount_rate=ctx["discount"],
                                    sizing_tolerance=float(cp["standalone_sizing_tolerance"]))
            if not sa["eligible"]:
                continue
            net = sc["network_avoided"][cl.cluster_id]
            gen_flat = _avoided_grid_cost(cl, flat_ctx, sc["grid"], include_losses=sc["losses"]) - net
            need = sa["total_npc"] - net
            rows.append({"cluster_id": cl.cluster_id, "n_customers": cl.n_customers,
                         "shs_npc": sa["total_npc"], "network_avoided": net,
                         "generation_credit_flat": gen_flat,
                         "critical_credit_share": (need / gen_flat) if gen_flat > 0 else None})
        sc["decomposition"] = rows
        net_alone = sum(1 for r in rows if (r["critical_credit_share"] or 0) <= 0)
        shares = [r["critical_credit_share"] for r in rows if (r["critical_credit_share"] or 0) > 0]
        print(f"\nScreen {name} decomposition: SHS beats the network alone on {net_alone}/{len(rows)} "
              f"branches; of the rest, the tightest needs {max(shares):.0%} of the flat generation "
              f"credit" if shares else f"\nScreen {name}: every branch beats the network alone", flush=True)

    results = {"meta": {"dataset": "mafa", "flat_lcoe": flat_lcoe, "horizon_years": T,
                        "discount_rate": float(cp["discount_rate"]), "max_rounds": MAX_ROUNDS,
                        "n_customers": int(len(inp["customers"]))},
               "runs": []}

    for name, sc in screens.items():
        n_cand = len(sc["network_avoided"])
        print(f"\nScreen {name}: {n_cand} eligible candidate branches over {len(sc['grid'])} grid customers")
        for level in LCOE_LEVELS:
            naive_off = None
            for fixed_share, alpha in SHAPES:
                model = ScaleEconomyGeneration.calibrated(fixed_share, alpha, level, sc["e_full"])
                for damping in DAMPING:
                    if (fixed_share, alpha) == (0.0, 1.0) and damping != 1.0:
                        continue  # the flat model is state-independent; damping is meaningless
                    r = run_loop(ctx, sc["clusters"], sc["network_avoided"], sc["grid"], model,
                                 damping, sc["label"], sc["cid"])
                    if (fixed_share, alpha) == (0.0, 1.0):
                        naive_off = set(r["final_off"])
                        if level == flat_lcoe:
                            # Harness check: the flat curve at the dataset's price must
                            # reproduce the production screen exactly.
                            assert naive_off == sc["production_off"], (
                                f"screen {name}: harness disagrees with production "
                                f"({len(naive_off)} vs {len(sc['production_off'])} off-grid)")
                    r.update({"screen": name, "lcoe_level": level, "fixed_share": fixed_share,
                              "alpha": alpha, "damping": damping,
                              "marginal_over_flat_at_full": alpha * (1 - fixed_share),
                              "naive_off": sorted(naive_off),
                              "differs_from_naive": set(r["final_off"]) != naive_off})
                    results["runs"].append(r)
                    last = r["rounds"][-1]
                    trail = " ".join(str(x["n_off_grid"]) for x in r["rounds"][:8])
                    print(f"  {level:.2f}EUR ({fixed_share:.2f},{alpha:.2f}) damp={damping:.1f}: "
                          f"{r['outcome']:<17s} {r['n_rounds']:>2} rounds -> {last['n_off_grid']:>3} off "
                          f"(flat {len(naive_off):>3})   off-grid by round: {trail}", flush=True)

    # Realised Case D, priced with the synthetic curve as "the truth". For each converged
    # shape, compare the coupled answer with the naive one (the flat-LCOE screen's off-grid
    # set, i.e. what the pipeline recommends today) under the SAME curve: that difference
    # is the value of coupling. The network rebuild does not depend on the curve, so it is
    # done once per distinct off-grid set and only the generation term is re-priced.
    print("\nRealised Case D under each curve: coupled vs naive (flat-screen) off-grid set ...", flush=True)
    no_gen_ctx = {**ctx, "generation_model": FlatLcoeGeneration(0.0)}
    built: Dict[tuple, Dict] = {}

    def realise(off_ids) -> Dict:
        key = tuple(sorted(off_ids))
        if key not in built:
            ctx["log"] = []
            sa = cluster_standalone(list(key), ctx["customers"], ctx["packages"], horizon_years=T,
                                    discount_rate=ctx["discount"],
                                    sizing_tolerance=float(cp["standalone_sizing_tolerance"]))
            res = _run_mvlv({**no_gen_ctx, "log": []},
                            [c for c in screens["D"]["grid"] if c not in set(key)],
                            cid="D", architecture="MV/LV + Stand-alone",
                            sa_npc=(sa["total_npc"] if key else 0.0), sa_customers=list(key))
            built[key] = {"feasible": bool(res.feasible), "npc_without_generation": float(res.cost.total_npc),
                          "n_transformers": int(res.n_transformers),
                          "e_served": levelised_energy(ctx, res.grid_customer_ids),
                          "n_grid": int(res.n_grid_customers), "n_off": len(key)}
            print(f"  built D with {len(key):>3} off-grid: {built[key]['n_transformers']} zones, "
                  f"{'feasible' if res.feasible else 'INFEASIBLE'}", flush=True)
        return built[key]

    for r in results["runs"]:
        if (r["screen"] != "D" or r["damping"] != 1.0 or r["outcome"] != "converged"
                or not r["differs_from_naive"]):
            continue
        model = ScaleEconomyGeneration.calibrated(r["fixed_share"], r["alpha"], r["lcoe_level"],
                                                  screens["D"]["e_full"])
        priced = {}
        for tag, off_ids in (("coupled", r["final_off"]), ("naive", r["naive_off"])):
            b = realise(off_ids)
            priced[tag] = {**b, "npc_total": b["npc_without_generation"] + model.cost(b["e_served"])}
        r["realised"] = priced
        r["coupling_saving"] = priced["naive"]["npc_total"] - priced["coupled"]["npc_total"]

    print("\n  level  shape (fixed, alpha) | off-grid coupled/naive | NPC coupled | NPC naive | coupling saves")
    for r in results["runs"]:
        if "realised" not in r:
            continue
        c, n = r["realised"]["coupled"], r["realised"]["naive"]
        print(f"  {r['lcoe_level']:.2f}   ({r['fixed_share']:.2f}, {r['alpha']:.2f})        | "
              f"{c['n_off']:>3} / {n['n_off']:>3}              | {c['npc_total']:>11,.0f} | "
              f"{n['npc_total']:>9,.0f} | {r['coupling_saving']:>+12,.0f}"
              f"{'' if c['feasible'] and n['feasible'] else '  (an infeasible build)'}", flush=True)

    results["decomposition"] = {name: sc["decomposition"] for name, sc in screens.items()}
    out = OUT / "convergence_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out.relative_to(ROOT)} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
