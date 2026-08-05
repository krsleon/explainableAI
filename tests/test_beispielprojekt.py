"""Tests der Analysefunktionen des Beispielprojekts (Project STAR).

Prüft die Zahlen, auf denen die Projektseite ihre Aussagen aufbaut. Läuft
ohne Streamlit, weil `analyse.py` bewusst UI-frei ist.
"""

import sys
from pathlib import Path

import pytest

PROJEKT = Path(__file__).resolve().parent.parent / "content" / "projekte" / "beispielprojekt"
sys.path.insert(0, str(PROJEKT))

analyse = pytest.importorskip("analyse")


@pytest.fixture(scope="module")
def daten():
    return analyse.lade_daten()


def test_daten_laden(daten):
    assert len(daten) == 6325
    assert set(daten["stark"].unique()) == set(analyse.ARME)
    assert "gesamt" in daten.columns


def test_gesamtscore_ist_mittel_aus_beiden_tests(daten):
    zeile = daten.dropna(subset=["readk", "mathk"]).iloc[0]
    assert zeile["gesamt"] == pytest.approx((zeile["readk"] + zeile["mathk"]) / 2)


def test_ueberblick_deckt_alle_arme_ab(daten):
    tabelle = analyse.ueberblick(daten)
    assert len(tabelle) == 3
    assert tabelle["Kinder"].sum() == len(daten)
    assert (tabelle["mit Testergebnis"] <= tabelle["Kinder"]).all()


def test_kleine_klassen_wirken_positiv(daten):
    """Der zentrale Befund: rund +7 Punkte, etwa 0,2 SD (vgl. Krueger 1999)."""
    ergebnis = analyse.ate_mit_ci(daten)
    klein = ergebnis[ergebnis["Vergleich"].str.startswith(analyse.ARM_LABEL["small"])]
    assert len(klein) == 1
    zeile = klein.iloc[0]
    assert 5 < zeile["Effekt (Punkte)"] < 10
    assert 0.15 < zeile["Effekt (SD)"] < 0.25
    assert zeile["KI unten"] > 0, "Konfidenzintervall soll die Null ausschließen"
    assert zeile["p-Wert"] < 0.01


def test_hilfskraft_zeigt_keinen_effekt(daten):
    """Nullbefund: Das KI enthält die Null und ist eng genug für eine Aussage."""
    ergebnis = analyse.ate_mit_ci(daten)
    aide = ergebnis[ergebnis["Vergleich"].str.startswith(analyse.ARM_LABEL["regular+aide"])]
    zeile = aide.iloc[0]
    assert zeile["KI unten"] < 0 < zeile["KI oben"]
    assert abs(zeile["Effekt (SD)"]) < 0.05
    assert zeile["p-Wert"] > 0.1


def test_schul_fixed_effects_praezisieren_ohne_zu_verschieben(daten):
    """Das Design verlangt School FE; der Punktschätzer darf kaum wandern."""
    mit = analyse.ate_mit_ci(daten, schul_fe=True).iloc[1]
    ohne = analyse.ate_mit_ci(daten, schul_fe=False).iloc[1]
    assert abs(mit["Effekt (Punkte)"] - ohne["Effekt (Punkte)"]) < 2
    breite_mit = mit["KI oben"] - mit["KI unten"]
    breite_ohne = ohne["KI oben"] - ohne["KI unten"]
    assert breite_mit < breite_ohne


def test_balance_tabelle_vollstaendig(daten):
    tabelle = analyse.balance_tabelle(daten)
    erwartet = {label for label, _ in analyse.VORAB_MERKMALE.values()}
    assert set(tabelle["Merkmal"]) == erwartet
    for arm in analyse.ARME:
        spalte = analyse.ARM_LABEL[arm]
        assert tabelle[spalte].between(0, 1).all()
    # Anteile summieren sich je Merkmal und Arm auf 1.
    for _, gruppe in tabelle.groupby("Merkmal"):
        for arm in analyse.ARME:
            assert gruppe[analyse.ARM_LABEL[arm]].sum() == pytest.approx(1.0)


def test_kindmerkmale_sind_balanciert(daten):
    """Das Los muss die Merkmale der *Kinder* gleich verteilen."""
    tabelle = analyse.balance_tabelle(daten)
    kind = tabelle[tabelle["Ebene"] == "Kind"]
    assert (kind["p-Wert (Merkmal)"] > 0.05).all()
    assert kind["Max. Differenz (%-Punkte)"].max() < 5


def test_schulmerkmal_ist_erwartungsgemaess_unbalanciert(daten):
    """Zugelost wurde innerhalb der Schule, deshalb muss der Schultyp über die
    gepoolten Arme *nicht* balancieren. Der Unterschied bleibt aber klein,
    signifikant wird er nur durch die große Stichprobe. Genau darauf baut die
    Erklärung im Identifikations-Tab auf."""
    tabelle = analyse.balance_tabelle(daten)
    schule = tabelle[tabelle["Ebene"] == "Schule"]
    assert not schule.empty
    assert (schule["p-Wert (Merkmal)"] < 0.05).all()
    assert schule["Max. Differenz (%-Punkte)"].max() < 8


def test_ausfaelle_sind_ueber_die_arme_gleichverteilt(daten):
    tabelle = analyse.ausfall_uebersicht(daten)
    spanne = tabelle["Anteil fehlend"].max() - tabelle["Anteil fehlend"].min()
    assert spanne < 0.02, "Ungleiche Attrition würde den Vergleich gefährden"


def test_heterogenitaet_nach_mittagessen(daten):
    tabelle = analyse.heterogenitaet(daten)
    assert set(tabelle["Teilgruppe"]) == {"free", "non-free"}
    # Kinder aus einkommensschwaecheren Haushalten profitieren staerker.
    frei = tabelle.loc[tabelle["Teilgruppe"] == "free", "Effekt (SD)"].iloc[0]
    rest = tabelle.loc[tabelle["Teilgruppe"] == "non-free", "Effekt (SD)"].iloc[0]
    assert frei > rest > 0


def test_erfahrungsvergleich_ist_groesser_als_der_kausale_effekt(daten):
    """Didaktischer Kern von Tab 3: die nicht randomisierte Variable sieht
    beeindruckender aus als die randomisierte."""
    mit_erfahrung = daten.copy()
    mit_erfahrung["erfahrung"] = analyse.erfahrungsgruppe(mit_erfahrung)
    roh = analyse.rohdifferenz(mit_erfahrung, "erfahrung", "0–5 Jahre")
    erfahren = roh.loc[roh["Gruppe"] == "über 15 Jahre", "Differenz zur Referenz"].iloc[0]

    kausal = analyse.ate_mit_ci(daten).iloc[1]["Effekt (Punkte)"]
    assert erfahren > kausal > 0


def test_verteilung_je_arm_liefert_rohwerte(daten):
    verteilungen = analyse.verteilung_je_arm(daten)
    assert set(verteilungen) == set(analyse.ARM_LABEL.values())
    assert all(len(werte) > 1000 for werte in verteilungen.values())
