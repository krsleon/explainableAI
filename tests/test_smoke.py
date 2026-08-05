"""Smoke-Tests: jede Seite der App rendert ohne Exception.

Nutzt streamlit.testing.v1.AppTest, kein Browser nötig.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
HAUPTSKRIPT = str(ROOT / "streamlit_app.py")

STATISCHE_SEITEN = [
    "views/start.py",
    "views/ueber.py",
    "views/ml/grundlagen.py",
    "views/ml/lineare_regression.py",
    "views/ml/regularisierung.py",
    "views/ml/baeume_ensembles.py",
    "views/ml/neuronale_netze.py",
    "views/ml/explainable_ml.py",
    "views/ml/llms_kausalitaet.py",
    "views/kausalitaet/korrelation.py",
    "views/kausalitaet/dags_confounding.py",
    "views/kausalitaet/potential_outcomes.py",
    "views/kausalitaet/quasi_experimente.py",
    "views/kausalitaet/kausales_ml.py",
    "views/kausalitaet/bayes.py",
    "views/kausalitaet/sem_surveys.py",
    "views/projekte/themen.py",
    "views/projekte/uebersicht.py",
    "views/referenzen.py",
]


def _app() -> AppTest:
    return AppTest.from_file(HAUPTSKRIPT, default_timeout=30)


def test_startseite_laedt():
    at = _app()
    at.run()
    assert not at.exception


@pytest.mark.parametrize("seite", STATISCHE_SEITEN)
def test_seite_rendert_ohne_fehler(seite):
    at = _app()
    at.switch_page(seite)
    at.run()
    assert not at.exception, f"{seite} wirft: {at.exception}"


def test_beispielprojekt_app_seite():
    at = _app()
    at.switch_page("content/projekte/beispielprojekt/app.py")
    at.run()
    assert not at.exception


def test_markdown_renderer_mit_vorlage():
    # AppTest.from_function kopiert nur den Funktions-Quelltext in ein
    # Temp-Skript; die Funktion muss daher selbstständig importieren.
    def seite(wurzel: str) -> None:
        import sys
        from pathlib import Path

        if wurzel not in sys.path:
            sys.path.insert(0, wurzel)
        from utils.projects import Projekt, render_markdown_projekt

        projekt = Projekt(
            slug="vorlage-test",
            titel="Vorlage",
            md_datei=Path(wurzel) / "content" / "projekte" / "_vorlage" / "projekt.md",
        )
        render_markdown_projekt(projekt)

    at = AppTest.from_function(seite, args=(str(ROOT),))
    at.run()
    assert not at.exception


def test_frontmatter_parsing():
    from utils.projects import _frontmatter

    meta, inhalt = _frontmatter("---\ntitel: Test\nemoji: \"🎯\"\n---\n\nHallo Welt")
    assert meta["titel"] == "Test"
    assert meta["emoji"] == "🎯"
    assert "Hallo Welt" in inhalt


def test_frontmatter_fehlt():
    from utils.projects import _frontmatter

    meta, inhalt = _frontmatter("Nur Text ohne Frontmatter")
    assert meta == {}
    assert inhalt == "Nur Text ohne Frontmatter"


def test_lade_projekte_findet_beispiel_und_ueberspringt_vorlage():
    from utils.projects import lade_projekte

    slugs = [p.slug for p in lade_projekte()]
    assert "beispielprojekt" in slugs
    assert all(not s.startswith("_") for s in slugs)

    beispiel = next(p for p in lade_projekte() if p.slug == "beispielprojekt")
    assert beispiel.app_datei is not None
    # Der Titel stammt aus dem Frontmatter, nicht aus dem Ordnernamen. Der
    # Fallback waere "Beispielprojekt" (slug, aufgehuebscht).
    assert beispiel.titel and beispiel.titel != "Beispielprojekt"
    assert beispiel.mitglieder
