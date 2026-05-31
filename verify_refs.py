"""
verify_refs.py
==============
Query the Crossref REST API for each reference in the manuscript, retrieve the
canonical title / first author / year / container / DOI, and write a report that
flags (a) the matched DOI and (b) any year/title mismatch to inspect.

Network: api.crossref.org (polite pool via mailto in User-Agent).
ASCII-only output.
"""
from __future__ import annotations
import urllib.request, urllib.parse, json, time, difflib

MAILTO = "omar.naifar@enis.tn"

# key -> (query string, expected_year, expected_title_fragment)
REFS = {
 "AguilaCamacho2014": ("Lyapunov functions for fractional order systems Aguila-Camacho Duarte-Mermoud Gallegos", 2014, "Lyapunov functions for fractional order systems"),
 "Akay2012": ("A modified artificial bee colony algorithm for real-parameter optimization Akay Karaboga", 2012, "modified artificial bee colony algorithm for real-parameter optimization"),
 "Akwasi2025": ("Second-order inertia automatic generation control grid-forming inverters convergent observers salp swarm frequency control", 2025, "Second-order inertia automatic generation control"),
 "Beck2019": ("Grid-forming inverters critical asset for the power grid Lasseter Chen Pattabiraman", 2020, "Grid-forming inverters"),
 "Blaabjerg2020": ("Control of Power Electronic Converters and Systems Blaabjerg", 2018, "Control of power electronic converters"),
 "Bubeck2015": ("Convex optimization algorithms and complexity Bubeck", 2015, "Convex Optimization: Algorithms and Complexity"),
 "Chen2009": ("Discretization schemes for fractional-order differentiators and integrators Chen Moore", 2002, "Discretization schemes for fractional-order differentiators and integrators"),
 "Doob1953": ("Stochastic Processes Doob", 1953, "Stochastic processes"),
 "DuarteMermoud2015": ("Using general quadratic Lyapunov functions to prove Lyapunov uniform stability for fractional order systems", 2015, "general quadratic Lyapunov functions"),
 "Dziri2025": ("Artificial bee colony optimization enhancement output power generation grid-connected photovoltaic systems Dziri Bouallegue", 2025, "Artificial bee colony"),
 "Guerrero2016": ("Advanced control architectures for intelligent microgrids Part I decentralized hierarchical control Guerrero", 2013, "Advanced control architectures for intelligent microgrids"),
 "Karaboga2007": ("A powerful and efficient algorithm for numerical function optimization artificial bee colony Karaboga Basturk", 2007, "powerful and efficient algorithm for numerical function optimization"),
 "Karimi2016": ("Linear convergence of gradient and proximal-gradient methods under the Polyak-Lojasiewicz condition Karimi Nutini Schmidt", 2016, "Polyak"),
 "Krause2002": ("Analysis of electric machinery and drive systems Krause Wasynczuk Sudhoff", 2002, "Analysis of electric machinery"),
 "Li2020": ("Remarks on fractional derivatives Li Deng", 2007, "Remarks on fractional derivatives"),
 "LiChenPodlubny2010": ("Stability of fractional-order nonlinear dynamic systems Lyapunov direct method generalized Mittag-Leffler stability", 2010, "Stability of fractional-order nonlinear dynamic systems"),
 "Matignon1996": ("Stability results for fractional differential equations with applications to control processing Matignon", 1996, "Stability results for fractional differential equations"),
 "Medved2021": ("Differential equations with tempered psi-Caputo fractional derivative Medved Brestovanska", 2021, "tempered"),
 "MeynTweedie2009": ("Markov chains and stochastic stability Meyn Tweedie", 2009, "Markov chains and stochastic stability"),
 "Mohan2003": ("Power electronics converters applications and design Mohan Undeland Robbins", 2003, "Power electronics"),
 "Monje2010": ("Fractional-order systems and controls fundamentals and applications Monje Chen Vinagre Xue Feliu", 2010, "Fractional-order systems and controls"),
 "Naifar2026": ("Tempered fractional gradient descent theory algorithms and robust learning applications Naifar", 2025, "Tempered fractional gradient descent"),
 "Nesterov2018": ("Lectures on convex optimization Nesterov", 2018, "Lectures on convex optimization"),
 "Oustaloup1995": ("Frequency-band complex noninteger differentiator characterization and synthesis Oustaloup Levron Mathieu Nanot", 2000, "Frequency-band complex noninteger differentiator"),
 "Podlubny1999": ("Fractional differential equations Podlubny", 1999, "Fractional differential equations"),
 "Pogaku2007": ("Modeling analysis and testing of autonomous operation of an inverter-based microgrid Pogaku Prodanovic Green", 2007, "autonomous operation of an inverter-based microgrid"),
 "Polyak1963": ("Gradient methods for the minimization of functionals Polyak", 1963, "Gradient methods for the minimization of functionals"),
 "RobbinsSiegmund1971": ("A convergence theorem for nonnegative almost supermartingales and some applications Robbins Siegmund", 1971, "almost supermartingales"),
 "Rocabert2012": ("Control of power converters in AC microgrids Rocabert Luna Blaabjerg Rodriguez", 2012, "Control of power converters in AC microgrids"),
 "SolisWets1981": ("Minimization by random search techniques Solis Wets", 1981, "Minimization by random search techniques"),
 "Sun2014": ("Impedance-based stability criterion for grid-connected inverters Sun", 2011, "Impedance-based stability criterion for grid-connected inverters"),
 "Valerio2005": ("Ninteger a non-integer control toolbox for MATLAB Valerio Sa da Costa", 2004, "Ninteger"),
 "Williams1991": ("Probability with martingales Williams", 1991, "Probability with martingales"),
 "Zhong2017": ("Power-electronics-enabled autonomous power systems architecture and technical routes Zhong", 2017, "Power-Electronics-Enabled Autonomous Power Systems"),
 "hadamard2025": ("Finite time stability for Hadamard fractional-order systems Naifar Ben Makhlouf Mchiri Rhaima", 2025, "Finite time stability for Hadamard fractional-order systems"),
 "barbalat2023": ("On the Barbalat lemma extension for the generalized conformable fractional integrals adaptive observer design", 2023, "Barbalat lemma extension"),
}


def crossref(query, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": rows})
    req = urllib.request.Request(url, headers={
        "User-Agent": "ref-verify/1.0 (mailto:%s)" % MAILTO})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)["message"]["items"]


def norm(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch == " ").strip()


def main():
    out = []
    for key, (q, yr, frag) in REFS.items():
        try:
            items = crossref(q)
        except Exception as e:
            out.append("%-20s QUERY_FAIL %s" % (key, e))
            time.sleep(1.0)
            continue
        best = None
        fragn = norm(frag)
        for it in items:
            t = (it.get("title") or [""])[0]
            score = difflib.SequenceMatcher(None, fragn, norm(t)).ratio()
            if fragn in norm(t):
                score = max(score, 0.95)
            if best is None or score > best[0]:
                best = (score, it)
        if not best:
            out.append("%-20s NO_RESULT" % key)
            time.sleep(1.0)
            continue
        score, it = best
        t = (it.get("title") or [""])[0]
        doi = it.get("DOI", "")
        # publication year
        dp = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        cyr = dp[0] if dp else None
        cont = (it.get("container-title") or [""])
        cont = cont[0] if cont else ""
        flag = ""
        if score < 0.55:
            flag += " [LOW_MATCH %.2f]" % score
        if cyr and yr and abs(int(cyr) - int(yr)) > 1:
            flag += " [YEAR exp=%s got=%s]" % (yr, cyr)
        out.append("%-20s DOI=%s | %sCR_year=%s | %s | %s%s" % (
            key, doi, "", cyr, t[:60], cont[:34], flag))
        time.sleep(0.8)
    open("ref_report.txt", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
    print("\nWROTE ref_report.txt (%d refs)" % len(REFS))


if __name__ == "__main__":
    main()
