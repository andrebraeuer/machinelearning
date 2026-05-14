"""
Calculates the Mean Absolute Deviation (MAD) across different VaR estimates from various copula models for each day.
"""

def calculate_mad(df: pd.DataFrame) -> pd.Series:
    """
    Berechnet den MAD.

    Parameter
    ----------
    df : pd.DataFrame
        Jede Spalte = Copula-Modell
        Jede Zeile = Zeitpunkt / Tag
        Werte = VaR-Schätzungen

    Returns
    -------
    pd.Series
        MAD pro Tag
    """
    # Durchschnittlicher VaR pro Tag
    row_mean = df.mean(axis=1)

    # Absolute Abweichungen vom Tagesmittel
    abs_dev = df.sub(row_mean, axis=0).abs()

    # Mean Absolute Deviation pro Tag
    mad = abs_dev.mean(axis=1)

    return mad


def mad_summary_statistics(
    mad_series: pd.Series,
    portfolio_value: float = 100_000,
    horizon_days: int = 10
) -> pd.DataFrame:
    """
    Berechnet die Summary-Statistiken des MAD wie in Table 2.

    Erwartet MAD-Werte als Dezimalzahlen:
    Beispiel:
        0.025 = 2.5%

    Returns
    -------
    pd.DataFrame
        Tabelle mit:
        - MAD (%)
        - MAD ($)
    """

    import numpy as np
    import pandas as pd

    # Statistiken aus Dezimalwerten
    stats_decimal = pd.Series({
        "Min": mad_series.min(),
        "Median": mad_series.median(),
        "Mean": mad_series.mean(),
        "Max": mad_series.max(),
        "SD": mad_series.std()
    })

    # In Prozent umrechnen
    stats_percent = stats_decimal * 100

    # Dollar-Skalierung wie im Paper
    scale = portfolio_value * np.sqrt(horizon_days)

    stats_dollar = stats_decimal * scale

    # Ergebnis
    result = pd.DataFrame({
        "MAD (%)": stats_percent,
        "MAD ($)": stats_dollar
    })

    return result