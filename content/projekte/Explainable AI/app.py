""" Explainable AI App
Frage: Wie koennen wir die Art von Pinguinen erkennen 
und die Entscheidung des KI-Modells nachvollziehen?"""

import sys
from pathlib import Path
import os

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
#============================================================= Intro to xAI
with tab_intro:
    st.markdown("## 1 · Einleitung: Explainable AI")
    st.caption(
        "Explainable AI (xAI) ist ein Forschungsgebiet, das sich mit der Nachvollziehbarkeit von Entscheidungen von KI-Modellen beschäftigt. "
        "Es geht darum, die Funktionsweise von Modellen zu verstehen und ihre Entscheidungen zu erklären."
        "Dazu gibt es verschiedene Herangehensweisen, die schematisch in der folgenden Abbildung dargestellt werden."
    )
    st.image(os.path.join(ORDNER, "images", "xAI_overview.png"), caption="Abbildung 1: Überblick über Explainable AI Methoden")
    #st.markdown(
    #"""
    #<div style="text-align: center;">
    #    <img src="images/xAI_overview.png" width="500">
    #</div>
    #""",
    #unsafe_allow_html=True
#)
    st.markdown(
        "In diesem Projekt werden wir uns auf zwei Methoden konzentrieren: LIME (Local Interpretable Model-agnostic Explanations) und SHAP (SHapley Additive exPlanations). "
        "Beide Methoden sind darauf ausgelegt, die Vorhersagen von Black-Box-Modellen zu erklären."
    )


#============================================================= Daten
with tab_daten:
    data = analyse.load_data()
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

with tab_blackbox:
    st.markdown(
        "Klassische, lineare Machine Learning Modelle bieten eine gewisse Interpretierbarkeit in ihren Entscheidungen, indem Koeffizienten und Verzweigungen direkt abzulesen sind."
    )
    st.markdown(
        "Bei hochdimensionalen Modellen hingegen leidet die Interpretierbarkeit unter der besseren Performance – die Millionen nicht-linear verknüpften Parameter entziehen sich der menschlichen Intuition."
    )
    st.markdown(
        "Aus diesem Grund spricht man von *Blackbox-Modellen*."
    )