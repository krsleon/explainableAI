""" Explainable AI App
Frage: Wie koennen wir die Art von Pinguinen erkennen 
und die Entscheidung des KI-Modells nachvollziehen?"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Der Projektordner liegt nicht automatisch im Suchpfad, weil Streamlit die
# Seite aus dem Hauptverzeichnis heraus startet. Diese drei Zeilen braucht
# ihr, sobald ihr eigenen Code aus Nachbardateien importieren wollt.
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

import analyse  # noqa: E402  (erst nach dem sys.path-Eintrag importierbar)
from utils.theming import FARBEN, merkkasten

st.markdown("# Explainable AI: Wie können wir Black-Box Modelle verstehen?")
st.caption(
    "Projekt von Katharina Gudat und Leon Kraus · Daten: Palmer Penguins, 2014, 344 Penguins"
)

merkkasten(
    "Dieses Projekt veranschaulicht zwei Modelle, die Black-Box Entscheidungen nachvollziehbar machen.",
    "Vor allem geht es um <b>Lime</b> und <b>SHAP</b>. "
    "Tab 1 ist eine Einleitung in Explainable AI, Tab 2 zeigt das Pinguin-Datenset, Tab 3 beschreibt das verwendete Blackbox-Modell, Tab 4 behandelt die LIME-Methode, Tab 5 die SHAP-Methode, und Tab 6 geht kurz auf andere Möglichkeiten der Explainable AI ein.",
    typ="merke",
)

tab_intro, tab_daten, tab_blackbox, tab_lime, tab_shap, tab_ausblick = st.tabs(
    [
        "1 · Einleitung",
        "2 · Daten: Palmer Penguins",
        "3 · Das Blackbox-Modell",
        "4 · LIME",
        "5 · SHAP",
        "6 · Ausblick: Andere Möglichkeiten",
    ]
)

#============================================================= Daten
with tab_daten:
    st.markdown("## 2 · Daten: Palmer Penguins")
    st.caption(
        "Die Daten stammen aus dem Datensatz 'Palmer Penguins' von Allison Horst, 2014. "
        "Er enthält Informationen zu 344 Pinguinen von drei Arten (Adelie, Chinstrap, Gentoo) "
        "und vier Inseln (Biscoe, Dream, Torgersen, und Palmer)."
    )
    st.markdown(
        "Die Daten enthalten die folgenden Spalten: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`."
    )
    st.markdown(
        "Wir werden die Spalten `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, und `body_mass_g` als Features verwenden, um die Art der Pinguine vorherzusagen."
    )
    