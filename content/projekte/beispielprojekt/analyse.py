"""Analysefunktionen für das Beispielprojekt (Project STAR).

Bewusst ohne Streamlit-Import: Hier steht *nur* die Analyse, das UI liegt in
`app.py`. Diese Trennung ist Absicht und Teil der Vorlage: So lassen sich
die Ergebnisse testen (siehe `tests/test_beispielprojekt.py`), ohne die App
zu starten, und ihr könnt dieselben Funktionen in einem Notebook benutzen.

Daten: Project STAR, Kindergarten-Kohorte. Siehe `daten/QUELLE.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATEN = Path(__file__).parent / "daten" / "star_kindergarten.csv"

# Die drei Versuchsarme des Experiments, in der Reihenfolge, in der wir sie
# überall anzeigen. "regular" ist die Referenzgruppe aller Vergleiche.
ARME = ["regular", "regular+aide", "small"]

ARM_LABEL = {
    "regular": "Normale Klasse (22–25 Kinder)",
    "regular+aide": "Normale Klasse + Hilfskraft",
    "small": "Kleine Klasse (13–17 Kinder)",
}

# Merkmale, die vor der Zuteilung feststanden, mit der Ebene, auf der sie
# gemessen sind. Die Unterscheidung ist wichtig: Zugelost wurde *innerhalb*
# jeder Schule, also müssen sich nur die Merkmale der Kinder über die Arme
# gleich verteilen. Schulmerkmale müssen das nicht, dazu mehr in
# `balance_tabelle`.
VORAB_MERKMALE = {
    "gender": ("Geschlecht", "Kind"),
    "ethnicity": ("Ethnische Zugehörigkeit", "Kind"),
    "lunchk": ("Anspruch auf kostenloses Mittagessen", "Kind"),
    "schoolk": ("Schultyp", "Schule"),
}

KIND_MERKMALE = [s for s, (_, e) in VORAB_MERKMALE.items() if e == "Kind"]


def lade_daten() -> pd.DataFrame:
    """Lädt die Kindergarten-Kohorte und ergänzt den Gesamt-Testscore.

    `gesamt` ist der Mittelwert aus Lese- und Mathematik-Score. Kinder, bei
    denen beide Tests fehlen, behalten NaN. Wir werfen sie hier bewusst
    *nicht* weg, damit die Ausfälle im EDA-Tab sichtbar bleiben.
    """
    daten = pd.read_csv(DATEN)
    daten["gesamt"] = daten[["readk", "mathk"]].mean(axis=1)
    daten["arm_label"] = daten["stark"].map(ARM_LABEL)
    return daten


def ueberblick(daten: pd.DataFrame) -> pd.DataFrame:
    """Fallzahlen und mittlere Testscores je Versuchsarm."""
    zeilen = []
    for arm in ARME:
        teil = daten[daten["stark"] == arm]
        werte = teil["gesamt"].dropna()
        zeilen.append(
            {
                "Versuchsarm": ARM_LABEL[arm],
                "Kinder": len(teil),
                "mit Testergebnis": len(werte),
                "Mittlerer Score": werte.mean(),
                "Standardabweichung": werte.std(),
            }
        )
    return pd.DataFrame(zeilen)


def balance_tabelle(daten: pd.DataFrame) -> pd.DataFrame:
    """Prüft, ob sich die Arme in Vorab-Merkmalen unterscheiden.

    Für jede Ausprägung eines Merkmals steht der Anteil innerhalb des Arms.
    Die Spalte „Max. Differenz“ ist die größte Abweichung zwischen zwei
    Armen in Prozentpunkten, ein Chi-Quadrat-Test liefert den p-Wert.

    Zur Lesart: Erwarten dürfen wir Gleichverteilung nur bei den Merkmalen
    der **Kinder**. Der Schultyp ist ein Merkmal der Schule, und zugelost
    wurde innerhalb jeder Schule. Weil kleine Klassen weniger Kinder fassen
    und die Schulen unterschiedlich viele Klassen je Arm beisteuerten,
    unterscheidet sich die Schulzusammensetzung der Arme in der gepoolten
    Stichprobe zwangsläufig ein wenig. Das ist kein Defekt der
    Randomisierung, sondern der Grund, warum School Fixed Effects in die
    Schätzung gehören.
    """
    from scipy import stats

    zeilen = []
    for spalte, (label, ebene) in VORAB_MERKMALE.items():
        kreuz = pd.crosstab(daten["stark"], daten[spalte])
        anteile = kreuz.div(kreuz.sum(axis=1), axis=0)
        p_wert = stats.chi2_contingency(kreuz)[1]
        for auspraegung in anteile.columns:
            eintrag = {
                "Ebene": ebene,
                "Merkmal": label,
                "Ausprägung": str(auspraegung),
            }
            for arm in ARME:
                eintrag[ARM_LABEL[arm]] = anteile.loc[arm, auspraegung]
            spannweite = anteile[auspraegung].max() - anteile[auspraegung].min()
            eintrag["Max. Differenz (%-Punkte)"] = 100 * spannweite
            eintrag["p-Wert (Merkmal)"] = p_wert
            zeilen.append(eintrag)
    return pd.DataFrame(zeilen)


def rohdifferenz(daten: pd.DataFrame, spalte: str, referenz: str) -> pd.DataFrame:
    """Mittelwertdifferenzen im Gesamtscore gegenüber einer Referenzgruppe.

    Rein deskriptiv, ohne jede Kontrolle. Genau das ist der Punkt: Die
    Funktion wird sowohl auf die randomisierte Klassengröße als auch auf die
    *nicht* randomisierte Lehrererfahrung angewendet, und liefert in beiden
    Fällen bereitwillig eine Zahl.
    """
    zeilen = []
    basis = daten.loc[daten[spalte] == referenz, "gesamt"].dropna()
    for gruppe in daten[spalte].dropna().unique():
        werte = daten.loc[daten[spalte] == gruppe, "gesamt"].dropna()
        zeilen.append(
            {
                "Gruppe": str(gruppe),
                "n": len(werte),
                "Mittelwert": werte.mean(),
                "Differenz zur Referenz": werte.mean() - basis.mean(),
            }
        )
    ergebnis = pd.DataFrame(zeilen).sort_values("Gruppe").reset_index(drop=True)
    return ergebnis


def erfahrungsgruppe(daten: pd.DataFrame) -> pd.Series:
    """Teilt die Lehrererfahrung in vier Bänder ein.

    Achtung: Anders als die Klassengröße wurde die Lehrererfahrung *nicht*
    zugelost. Die Einteilung dient nur dem Vergleich im Tab „Naive Analyse“.
    """
    return pd.cut(
        daten["experiencek"],
        bins=[-0.01, 5, 10, 15, 100],
        labels=["0–5 Jahre", "6–10 Jahre", "11–15 Jahre", "über 15 Jahre"],
    )


def ate_mit_ci(
    daten: pd.DataFrame,
    outcome: str = "gesamt",
    schul_fe: bool = True,
) -> pd.DataFrame:
    """Schätzt den Effekt der Versuchsarme gegenüber „regular“ per OLS.

    `schul_fe=True` nimmt Schul-Dummies auf. Das ist hier kein
    Confounding-Fix, sondern folgt dem Design: Zugelost wurde *innerhalb*
    jeder Schule, also gehört der Schulvergleich nicht in den Effekt. Die
    Schätzung wird dadurch präziser, der Punktschätzer bleibt ähnlich.

    Standardfehler sind auf Schulebene geclustert, weil die Zuteilung
    klassenweise erfolgte und Kinder derselben Schule korrelierte Ergebnisse
    haben.
    """
    teil = daten.dropna(subset=[outcome, "stark", "schoolidk"]).copy()
    teil["stark"] = pd.Categorical(teil["stark"], categories=ARME)

    formel = f"{outcome} ~ C(stark)"
    if schul_fe:
        formel += " + C(schoolidk)"

    modell = smf.ols(formel, data=teil).fit(
        cov_type="cluster", cov_kwds={"groups": teil["schoolidk"]}
    )

    streuung = teil[outcome].std()
    zeilen = []
    for arm in ARME[1:]:
        name = f"C(stark)[T.{arm}]"
        unten, oben = modell.conf_int().loc[name]
        zeilen.append(
            {
                "Vergleich": f"{ARM_LABEL[arm]} vs. {ARM_LABEL['regular']}",
                "Effekt (Punkte)": modell.params[name],
                "KI unten": unten,
                "KI oben": oben,
                "p-Wert": modell.pvalues[name],
                "Effekt (SD)": modell.params[name] / streuung,
            }
        )
    return pd.DataFrame(zeilen)


def heterogenitaet(
    daten: pd.DataFrame,
    nach: str = "lunchk",
    outcome: str = "gesamt",
) -> pd.DataFrame:
    """Schätzt den Effekt kleiner Klassen getrennt nach einer Teilgruppe.

    Erster Schritt Richtung CATE: Wirkt die Behandlung überall gleich? Die
    Teilgruppen sind vorab festgelegt (hier der Anspruch auf kostenloses
    Mittagessen als Indikator für den sozioökonomischen Hintergrund). Wer
    nachträglich so lange sucht, bis eine Teilgruppe signifikant wird,
    betreibt kein Lernen, sondern p-Hacking.
    """
    zeilen = []
    for auspraegung in sorted(daten[nach].dropna().unique()):
        teil = daten[daten[nach] == auspraegung]
        ergebnis = ate_mit_ci(teil, outcome=outcome, schul_fe=True)
        klein = ergebnis[ergebnis["Vergleich"].str.startswith(ARM_LABEL["small"])]
        zeile = klein.iloc[0].to_dict()
        zeile["Teilgruppe"] = str(auspraegung)
        zeile["n"] = len(teil.dropna(subset=[outcome]))
        zeilen.append(zeile)
    spalten = ["Teilgruppe", "n", "Effekt (Punkte)", "KI unten", "KI oben", "p-Wert", "Effekt (SD)"]
    return pd.DataFrame(zeilen)[spalten]


def ausfall_uebersicht(daten: pd.DataFrame) -> pd.DataFrame:
    """Fehlende Testergebnisse je Versuchsarm.

    Relevant für die Limitationen: Wenn in einem Arm systematisch andere
    Kinder fehlen, ist die Vergleichbarkeit trotz Randomisierung bedroht.
    """
    zeilen = []
    for arm in ARME:
        teil = daten[daten["stark"] == arm]
        fehlend = teil["gesamt"].isna().sum()
        zeilen.append(
            {
                "Versuchsarm": ARM_LABEL[arm],
                "Kinder": len(teil),
                "ohne Testergebnis": int(fehlend),
                "Anteil fehlend": fehlend / len(teil),
            }
        )
    return pd.DataFrame(zeilen)


def verteilung_je_arm(daten: pd.DataFrame, outcome: str = "gesamt") -> dict[str, np.ndarray]:
    """Rohwerte je Arm, aufbereitet für Histogramme in der App."""
    return {
        ARM_LABEL[arm]: daten.loc[daten["stark"] == arm, outcome].dropna().to_numpy()
        for arm in ARME
    }
