"""
run_all.py
==========
One-shot reproducibility driver: regenerates every figure and every number used
in the revised manuscript.  Each stage is isolated so a failure in one does not
lose the others; full tracebacks are logged.  ASCII-only output.

    PYTHONUTF8=1 python run_all.py
"""
from __future__ import annotations
import json, os, traceback, time

os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
np.seterr(all="ignore")

SUMMARY = {}


def stage(name, fn):
    print("\n========== %s ==========" % name, flush=True)
    t0 = time.time()
    try:
        fn()
        print("[STAGE OK] %s  (%.1fs)" % (name, time.time() - t0), flush=True)
        return True
    except Exception:
        print("[STAGE FAIL] %s" % name, flush=True)
        traceback.print_exc()
        return False


def run_theory():
    import theory_validation as tv
    tv.main()


def run_gfi():
    import run_gfi as rg
    rg.main()


def merge_results():
    out = {}
    for f in ("theory_results.json", "gfi_results.json"):
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                out[f.replace(".json", "")] = json.load(fh)
    with open("all_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    # ASCII summary
    print("\n========== KEY NUMBERS ==========", flush=True)
    th = out.get("theory_results", {})
    if "lemma1_fitted_slopes" in th:
        print("Lemma 1 fitted vs predicted slopes:")
        for k in th["lemma1_fitted_slopes"]:
            print("   %s : fit=%.3f  pred=%.3f"
                  % (k, th["lemma1_fitted_slopes"][k],
                     th["lemma1_predicted_slopes"][k]))
    if "lemma2" in th:
        print("Lemma 2 min slack = %.3e  holds=%s"
              % (th["lemma2"]["min_slack"], th["lemma2"]["holds"]))
    if "theorem2_final_success" in th:
        print("Theorem 2 final success prob:", th["theorem2_final_success"])
    gf = out.get("gfi_results", {})
    if "tuning_best_cost" in gf:
        print("FOPID tuning best cost:", gf["tuning_best_cost"])
    if "step_response" in gf:
        print("Step response:", gf["step_response"])
    if "thd" in gf:
        print("THD:", gf["thd"])
    if "observer" in gf:
        print("Observer MAE:", gf["observer"])


def main():
    ok1 = stage("THEORY VALIDATION", run_theory)
    ok2 = stage("GFI SIMULATION", run_gfi)
    stage("MERGE RESULTS", merge_results)
    print("\n[ALL DONE]  theory=%s  gfi=%s" % (ok1, ok2), flush=True)
    figs = sorted(os.listdir("figures")) if os.path.exists("figures") else []
    print("figures (%d): %s" % (len(figs), ", ".join(figs)), flush=True)


if __name__ == "__main__":
    main()
