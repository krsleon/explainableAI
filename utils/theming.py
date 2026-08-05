"""Zentrales Theming: Farbpalette, Plotly-Template, CSS und UI-Bausteine.

Alle Seiten holen sich Farben und Bausteine von hier, damit die App
durchgehend einheitlich aussieht.
"""

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent

# Alpin-Palette (siehe assets/styles.css)
FARBEN = {
    "nacht": "#1E3A5F",
    "gletscher": "#3E6DB5",
    "himmel": "#7FA8D9",
    "sonne": "#E8804C",
    "wiese": "#4C956C",
    "beere": "#B23A48",
    "flieder": "#8A6FBF",
    "schiefer": "#5C6470",
    "nebel": "#F2F5FA",
}

COLORWAY = [
    FARBEN["gletscher"],
    FARBEN["sonne"],
    FARBEN["wiese"],
    FARBEN["beere"],
    FARBEN["flieder"],
    FARBEN["schiefer"],
]


def register_plotly_template() -> None:
    """Registriert das Plotly-Template "leysin" und setzt es als Default."""
    vorlage = go.layout.Template(
        layout=go.Layout(
            colorway=COLORWAY,
            font=dict(family="'Source Sans Pro', sans-serif", color="#22252A", size=14),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E3E8F0", zerolinecolor="#C9D2E0"),
            yaxis=dict(gridcolor="#E3E8F0", zerolinecolor="#C9D2E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=48, b=10),
        )
    )
    pio.templates["leysin"] = vorlage
    pio.templates.default = "plotly_white+leysin"


def inject_css() -> None:
    """Lädt assets/styles.css in die laufende Seite."""
    css = (_ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def kapitel_kopf(emoji: str, titel: str, untertitel: str) -> None:
    """Einheitlicher Kapitelkopf: Titel mit Emoji und Untertitel-Zeile."""
    st.markdown(
        f'<div class="kapitel-kopf"><h1>{emoji} {titel}</h1>'
        f"<p>{untertitel}</p></div>",
        unsafe_allow_html=True,
    )


def merkkasten(titel: str, text: str, typ: str = "merke") -> None:
    """Farbiger Kasten für Kernaussagen.

    typ: "merke" (grün), "definition" (blau), "achtung" (orange),
    "beispiel" (grau). `text` wird als HTML gerendert, kein Markdown;
    für Hervorhebungen <b>…</b> / <i>…</i> verwenden.
    """
    st.markdown(
        f'<div class="merkkasten {typ}"><div class="mk-titel">{titel}</div>{text}</div>',
        unsafe_allow_html=True,
    )


def gruppen_aufgabe(titel: str, fragen: list[str], hinweis: str = "") -> None:
    """Kasten mit offenen Fragen, die dieses Kapitel bewusst nicht beantwortet.

    Markiert die Grenze zwischen „das erklären wir“ und „das erarbeitet eure
    Gruppe“. `titel` und `hinweis` werden als HTML gerendert, ebenso die
    Einträge in `fragen`. Für Hervorhebungen <b>…</b> / <i>…</i> verwenden.
    """
    punkte = "".join(f"<li>{frage}</li>" for frage in fragen)
    hinweis_html = f'<div class="ga-hinweis">{hinweis}</div>' if hinweis else ""
    st.markdown(
        f'<div class="gruppen-aufgabe"><div class="ga-titel">{titel}</div>'
        f"<ul>{punkte}</ul>{hinweis_html}</div>",
        unsafe_allow_html=True,
    )


def vertiefung(titel: str, offen: bool = False):
    """Einheitlich beschrifteter Expander für vertiefende Abschnitte.

    Als Context Manager verwenden::

        with vertiefung("Wie SHAP-Werte entstehen"):
            st.markdown(...)

    Die Kapitel zeigen im Fließtext nur den Kern. Alles, was eine Gruppe
    selbst erarbeiten soll, wandert hier hinein.
    """
    return st.expander(f"Vertiefung: {titel}", expanded=offen)


def stub_seite(emoji: str, titel: str, punkte: list[str]) -> None:
    """Platzhalterseite für Kapitel, die während der Akademie entstehen."""
    kapitel_kopf(emoji, titel, "Dieses Kapitel entsteht während der Akademie.")
    st.markdown('<span class="badge-wip">🚧 In Arbeit</span>', unsafe_allow_html=True)
    st.markdown("### Das erwartet dich")
    for punkt in punkte:
        st.markdown(f"- {punkt}")
    st.divider()
    st.page_link("views/start.py", label="Zurück zur Startseite", icon="🏔️")
