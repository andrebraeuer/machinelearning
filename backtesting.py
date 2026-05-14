"""
Backtests for VaR and ES forecasts.
 
Implements two backtests for risk measure forecasts:
* Unconditional coverage test (Kupiec test) for VaR forecasts.
* Independence test (Christoffersen test) for VaR forecasts.


--- Christoffersen Independence Test ---
Der Christoffersen Independence Test prüft nur, ob die VaR-Verletzungen über die Zeit unabhängig sind, also ob es kein Clustering gibt.
Modelliert wird das über eine 2-Zustands-Markov-Kette: man schätzt die Übergangswahrscheinlichkeiten π₀₁ (Verletzung nach Nicht-Verletzung) 
und π₁₁ (Verletzung nach Verletzung) und testet, ob π₀₁ = π₁₁. Wenn ja → unabhängig. 
Die Höhe der Verletzungsrate ist diesem Test völlig egal — du könntest 20% Verletzungen bei einem 5%-VaR haben, 
und solange sie schön gleichmäßig verteilt sind, besteht der Test. LR ist χ²-verteilt mit 1 Freiheitsgrad


--- Christoffersen CC Test ---
Der Christoffersen Conditional Coverage Test kombiniert den Kupiec Test (über die Verletzungsrate) mit dem Christoffersen Independence Test
(über die zeitliche Unabhängigkeit der Verletzungen), d.h. prüft beides gleichzeitig:
1. die durchschnittliche Verletzungsrate entspricht α (das ist der Kupiec-/Unconditional-Coverage-Teil), und
2. die Verletzungen sind unabhängig über die Zeit (Independence-Teil).
Somit ist der CC-Test ist ein gemeinsamer Test beider Hypothesen.
Wenn er besteht, weißt du, dass dein Modell sowohl die richtige Rate als auch Unabhängigkeit liefert 
— Kupiec und Independence würden dann einzeln auch bestehen. Insofern ist er als „Pass/Fail-Gesamturteil" ausreichend


--- CC-Test als Hauptbacktest ---
Der CC-Test in der Praxis meistens der „Hauptbacktest" — 
er fällt nur durch, wenn das Modell entweder die falsche Häufigkeit oder die falsche zeitliche Struktur produziert.
Wenn der CC-Test ablehnt, weißt du nicht warum. 
Liegt's an einer falschen durchschnittlichen Verletzungsrate (Modell ist generell zu optimistisch/pessimistisch) oder 
am Clustering (Modell reagiert zu langsam auf Volatilitätsänderungen)?


--- Duration based test of Christoffersen and Pelletier (2004) ---
The test follows a more general approach of independence rather than focusing on the independence of VaR violations only. 
The test assumes that in the case of independent VaR violations, the time between violations must be independent 
of the time to a previous violation. In simple terms, this means that the probability that a VaR violation will occur in the next 10 days
must be independent of whether the last one occurred in the last 10 or 100 days (Fritzsch paper)

Der Duration-Based Test prüft — wie der Christoffersen-Independence-Test — ob VaR-Verletzungen unabhängig über die Zeit sind, a
lso ob es Clustering gibt. Aber er macht das über einen anderen Blickwinkel, der oft trennschärfer ist.

Grundidee: Statt sich Übergänge zwischen aufeinanderfolgenden Tagen anzuschauen, betrachtet der Test die Dauern zwischen Verletzungen 
— also: wie viele Tage liegen zwischen Hit 1 und Hit 2, zwischen Hit 2 und Hit 3, usw.

Intuition: Wenn dein VaR-Modell korrekt ist und Verletzungen wirklich zufällig auftreten, 
dann sollten die Wartezeiten zwischen Verletzungen einem ganz bestimmten Muster folgen — sie sollten exponentialverteilt sein. 
Das ist mathematisch die „memoryless property": Die Wahrscheinlichkeit, dass morgen eine Verletzung kommt, 
hängt nicht davon ab, wann die letzte war.

Wie der Test das prüft: 
1. Dauern berechnen: Aus dem Hit-Sequence-Vektor (0en und 1en) werden die Abstände zwischen den 1en extrahiert. 
Bei z.B. Verletzungen an Tag 5, 12, 13, 50 ergibt das Dauern von 7, 1, 37.

2. Zwei Modelle anpassen:
H₀ (Exponential): Dauern sind memorylos → Verletzungen sind unabhängig.
H₁ (Weibull): Dauern folgen einer flexibleren Weibull-Verteilung mit Shape-Parameter b.

3. Likelihood-Ratio-Test: Beide Modelle werden per Maximum Likelihood geschätzt und verglichen. 
Die Exponentialverteilung ist genau der Spezialfall der Weibull mit b=1b = 1
b=1, also kann man sauber testen, ob bb
b signifikant von 1 abweicht.

Was der Shape-Parameter b bedeutet. Das ist eigentlich das interessanteste Ergebnis des Tests:
- b<1: Clustering -> Kurz nach einer Verletzung ist die nächste wahrscheinlicher → Hazard-Rate fällt mit der Zeit
- b=0: Independence -> Exponentialverteilung, memorylos
- b>1: zu regelmäßige Verletzungen; Was passiert: Hazard-Rate steigt mit der Zeit
Der typische Fehlerfall bei VaR-Modellen ist b<1b: das Modell reagiert zu langsam auf Volatilitätsschocks, 
und nach einer Verletzung kommen schnell weitere hinterher (denk an Krisen — wenn ein Tag schlecht ist, sind die nächsten oft auch schlecht).


--- Duration based test of Christoffersen and Pelletier (2004) vs. Indepedence-Test ---
Beide testen Unabhängigkeit, aber:

- Christoffersen 1998 (Markov) schaut nur auf Übergänge von Tag zu Tag — also ob ein Hit gestern 
die Wahrscheinlichkeit eines Hits heute beeinflusst. Er fängt nur Clustering bei „Lag 1" ein.

- Christoffersen & Pelletier 2004 (Duration) schaut auf alle Zeitabstände. Wenn Verletzungen z.B. in Wellen alle 5 Tage kommen, 
sieht das der Markov-Test nicht (weil aufeinanderfolgende Tage unauffällig sind), der Duration-Test aber schon.

--> In der Literatur gilt der Duration-Test als deutlich powerful — er entdeckt Clustering, das der einfachere Markov-Test übersieht.
"""

def get_exceedances(realized, var_forecast):
    """
    Computes the hit sequence of VaR exceedances, i.e. a binary array where 1 indicates that the realized return was 
    less than the forecasted VaR (i.e. an exceedance), and 0 otherwise.

    Get binary array of exceedances (1 if return < VaR forecast, else 0).
    
    Parameters:
        - realized: Array-like, the realized returns.
        - var_forecast: Array-like, the forecasted VaR values.
    Returns:
        - Array of 0s and 1s indicating whether each return was an exceedance of the VaR forecast.

    """
    return (realized < var_forecast).astype(int)


def kupiec_pof_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Kupiec Proportion-of-Failures Test (unconditional coverage).
    H0: Die tatsächliche Verletzungsrate entspricht dem erwarteten Niveau alpha.

    Parameters:
        - violations: Array-like, binary array where 1 indicates a VaR exceedance (realized return < VaR forecast) and 0 otherwise.
        - alpha: Float, the theoretical violation rate (e.g., 0.05 for 5% VaR).
    Returns:
        - Dictionary containing:
            - n: Total number of observations.
            - violations: Total number of exceedances (sum of the violations array).
            - expected: Expected number of exceedances under the null hypothesis (n * alpha).
            - vr: Empirical violation rate (violations / n).
            - LR: Likelihood ratio test statistic for the Kupiec test.
            - p_value: p-value for the Kupiec test statistic.
    """
    import numpy as np
    from scipy import stats

    n = len(violations)
    x = int(violations.sum())
    if n == 0:
        return {"n": 0, "violations": 0, "expected": 0, "vr": np.nan, "LR": np.nan, "p_value": np.nan}

    pi_hat = x / n
    # Edge cases: keine oder nur Verletzungen
    if x == 0:
        ll1 = 0.0
    elif x == n:
        ll1 = 0.0
    else:
        ll1 = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)

    LR = -2.0 * (ll0 - ll1)
    p = 1.0 - stats.chi2.cdf(LR, df=1)
    return {"n": n, "violations": x, "expected": n * alpha, "vr": pi_hat, "LR": float(LR), "p_value": float(p)}


def christoffersen_independence_test(violations: np.ndarray) -> dict:
    """
    Christoffersen (1998) Independence Test via 2-State Markov chain.

    H0: Die Verletzungen sind unabhängig über die Zeit (keine Clusterbildung von Verletzungen).

    Parameters:
        - violations: Array-like, binary array where 1 indicates a VaR exceedance (realized return < VaR forecast) and 0 otherwise.
    Returns:
        - Dictionary containing:
            - LR: Likelihood ratio test statistic for the Christoffersen independence test.
            - p_value: p-value for the Christoffersen independence test statistic.
            - n00: Count of transitions from 0 to 0 (no violation followed by no violation).
            - n01: Count of transitions from 0 to 1 (no violation followed by violation).
            - n10: Count of transitions from 1 to 0 (violation followed by no violation).
            - n11: Count of transitions from 1 to 1 (violation followed by violation).
    """
    import numpy as np
    from scipy import stats

    v = np.asarray(violations).astype(int)
    n = len(v)
    if n < 2:
        return {"LR": np.nan, "p": np.nan}

    # Transition counts
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    # Transition probabilities (geschützt gegen 0/0)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Log-Likelihoods
    def _safe_log(p): return np.log(p) if p > 0 else 0.0

    ll_indep = ((n01 + n11) * _safe_log(pi)
                + (n00 + n10) * _safe_log(1 - pi))
    ll_markov = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
                 + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))

    LR = -2.0 * (ll_indep - ll_markov)
    p = 1.0 - stats.chi2.cdf(LR, df=1)
    return {"LR": float(LR), "p_value": float(p),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_cc_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Conditional Coverage Test (Kupiec + Independence kombiniert).
    H0: Die Verletzungen haben die richtige durchschnittliche Rate (alpha) UND sind unabhängig über die Zeit.
    Parameters:
    - violations: Array-like, binary array where 1 indicates a VaR exceedance (realized return < VaR forecast) and 0 otherwise.
    - alpha: Float, the theoretical violation rate (e.g., 0.05 for 5% VaR).
    Returns:
    - Dictionary containing:
        - LR: Likelihood ratio test statistic for the conditional coverage test.
        - p_value: p-value for the conditional coverage test statistic.
    """
    import numpy as np
    from scipy import stats

    uc = kupiec_pof_test(violations, alpha)
    ind = christoffersen_independence_test(violations)
    if np.isnan(uc["LR"]) or np.isnan(ind["LR"]):
        return {"LR": np.nan, "p_value": np.nan}
    LR_cc = uc["LR"] + ind["LR"]
    p = 1.0 - stats.chi2.cdf(LR_cc, df=2)
    return {"LR": float(LR_cc), "p_value": float(p)}


def run_backtests(var_df: pd.DataFrame,
                  realized: pd.Series,
                  alpha: float) -> pd.DataFrame:
    """
    Backtests für alle Modelle in `var_df` zum Niveau `alpha`.

    Parameters
    ----------
    var_df   : DataFrame mit VaR-Forecasts (Spalten = Modelle).  In deinem
               Setup sind das Quantile, also typischerweise NEGATIVE Zahlen
               für Verluste.  Eine Verletzung ist  r_t < VaR_t.
    realized : Realisierte Portfolio-Returns (gleicher Index wie var_df).
    alpha    : Theoretisches Verletzungsniveau (z.B. 0.05 oder 0.01).
    """
    import pandas as pd

    rows = []
    for model in var_df.columns:
        var_series = var_df[model].astype(float)
        viol = (realized.values < var_series.values).astype(int)

        uc  = kupiec_pof_test(viol, alpha)
        ind = christoffersen_independence_test(viol)
        cc  = christoffersen_cc_test(viol, alpha)

        rows.append({
            "Model":              model,
            "Alpha":              alpha,
            "N":                  uc["n"],
            "Violations":         uc["violations"],
            "Expected":           round(alpha * uc["n"], 2),
            "Violation Rate":     uc["vr"],
            "Kupiec LR":          uc["LR"],
            "Kupiec p_value":           round(uc["p_value"], 4),
            "Independence LR":    ind["LR"],
            "Independence p_value":     round(ind["p_value"], 4),
            "CC LR":              cc["LR"],
            "CC p_value":               round(cc["p_value"], 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def christoffersen_pelletier_duration_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen & Pelletier (2004) Duration-Based Test of Independence.

    H0: Die Dauern zwischen aufeinanderfolgenden VaR-Verletzungen sind exponentialverteilt
        mit Parameter alpha (d.h. memorylos / keine Clusterbildung).
    H1: Die Dauern folgen einer Weibull-Verteilung mit Shape-Parameter b != 1
        (b < 1 => Clustering, b > 1 => zu regelmäßige Verletzungen).

    Die Exponentialverteilung ist ein Spezialfall der Weibull-Verteilung mit b = 1.
    Der Test ist ein Likelihood-Ratio-Test mit 1 Freiheitsgrad.

    Parameters:
        - violations: Array-like, binary array where 1 indicates a VaR exceedance
                      (realized return < VaR forecast) and 0 otherwise.
        - alpha: Float, the theoretical violation rate (e.g., 0.05 for 5% VaR).
                 Wird nur informativ zurückgegeben; die MLE schätzt a und b frei.
    Returns:
        - Dictionary containing:
            - LR: Likelihood ratio test statistic.
            - p_value: p-value (chi-squared with 1 df).
            - b_hat: Geschätzter Weibull-Shape-Parameter (b=1 unter H0).
            - a_hat: Geschätzter Weibull-Scale-Parameter unter H1.
            - n_durations: Anzahl der berechneten Dauern.
            - n_violations: Gesamtanzahl der Verletzungen.
    """
    import numpy as np
    from scipy import stats, optimize

    v = np.asarray(violations).astype(int)
    n = len(v)
    n_viol = int(v.sum())

    # Mindestens 2 Verletzungen nötig, damit es mindestens eine "vollständige" Dauer gibt
    if n_viol < 2:
        return {"LR": np.nan, "p_value": np.nan, "b_hat": np.nan,
                "a_hat": np.nan, "n_durations": 0, "n_violations": n_viol}

    # Indizes der Verletzungen
    hit_idx = np.where(v == 1)[0]

    # Dauern zwischen aufeinanderfolgenden Verletzungen (no-hit Tage + 1)
    # D_i = t_i - t_{i-1}  (Anzahl Tage zwischen Hit i-1 und Hit i)
    durations = np.diff(hit_idx).astype(float)

    # Zensierte Dauern an den Rändern:
    # - vor der ersten Verletzung: hit_idx[0] + 1 Tage bis zur ersten Verletzung
    # - nach der letzten Verletzung: n - 1 - hit_idx[-1] Tage ohne weitere Verletzung
    d_first = float(hit_idx[0] + 1)            # links zensiert (kein vorheriger Hit beobachtet)
    d_last  = float(n - 1 - hit_idx[-1])       # rechts zensiert (kein nachfolgender Hit beobachtet)

    # Wir behandeln die Randdauern als rechtszensiert: wir wissen nur, dass D >= d_first
    # bzw. D >= d_last. (Standardvorgehen in Christoffersen & Pelletier 2004.)
    # Vollständig beobachtete Dauern:
    d_obs = durations
    # Zensierte Dauern (nur falls > 0):
    d_cens = []
    if d_first > 0:
        d_cens.append(d_first)
    if d_last > 0:
        d_cens.append(d_last)
    d_cens = np.array(d_cens, dtype=float)

    # ---------- Log-Likelihoods ----------
    # Weibull-Dichte (parametrisiert wie in C&P 2004):
    #   f(d; a, b) = a * b * (a*d)^(b-1) * exp(-(a*d)^b)
    #   S(d; a, b) = exp(-(a*d)^b)    (Survivor)
    # Exponentialverteilung: b = 1 => f(d; a) = a * exp(-a*d), S(d; a) = exp(-a*d)
    # MLE für Exponential mit Zensierung: a_hat = (#vollst. beobachtungen) / (Summe aller d)
    #
    # H0 (Exponential): wir schätzen a frei (oder fixieren a = alpha — siehe Hinweis unten).
    # H1 (Weibull): wir schätzen a und b frei.

    def neg_ll_weibull(params, d_obs, d_cens):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        # Beitrag vollständiger Beobachtungen: log f(d; a, b)
        ll_obs = np.sum(np.log(a) + np.log(b) + (b - 1) * np.log(a * d_obs) - (a * d_obs) ** b)
        # Beitrag zensierter Beobachtungen: log S(d; a, b) = -(a*d)^b
        if len(d_cens) > 0:
            ll_cens = -np.sum((a * d_cens) ** b)
        else:
            ll_cens = 0.0
        return -(ll_obs + ll_cens)

    def neg_ll_exp(a, d_obs, d_cens):
        if a <= 0:
            return 1e10
        ll_obs = np.sum(np.log(a) - a * d_obs)
        if len(d_cens) > 0:
            ll_cens = -a * np.sum(d_cens)
        else:
            ll_cens = 0.0
        return -(ll_obs + ll_cens)

    # ---------- H0: Exponential MLE ----------
    # Geschlossene Form mit Zensierung: a_hat = (#vollst. beobachtungen) / (Summe aller Dauern)
    total_time = float(d_obs.sum() + d_cens.sum())
    if total_time <= 0:
        return {"LR": np.nan, "p_value": np.nan, "b_hat": np.nan,
                "a_hat": np.nan, "n_durations": len(d_obs), "n_violations": n_viol}
    a_hat_h0 = len(d_obs) / total_time
    ll_h0 = -neg_ll_exp(a_hat_h0, d_obs, d_cens)

    # ---------- H1: Weibull MLE ----------
    # Startwerte: a = a_hat_h0, b = 1
    res = optimize.minimize(
        neg_ll_weibull,
        x0=[a_hat_h0, 1.0],
        args=(d_obs, d_cens),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000}
    )
    if not res.success:
        return {"LR": np.nan, "p_value": np.nan, "b_hat": np.nan,
                "a_hat": np.nan, "n_durations": len(d_obs), "n_violations": n_viol}
    a_hat_h1, b_hat_h1 = res.x
    ll_h1 = -res.fun

    # ---------- LR-Test ----------
    LR = -2.0 * (ll_h0 - ll_h1)
    # Numerischer Schutz: LR sollte >= 0 sein
    LR = max(LR, 0.0)
    p_value = 1.0 - stats.chi2.cdf(LR, df=1)

    return {
        "LR": float(LR),
        "p_value": float(p_value),
        "b_hat": float(b_hat_h1),
        "a_hat": float(a_hat_h1),
        "n_durations": len(d_obs),
        "n_violations": n_viol,
    }