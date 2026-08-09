"""Über uns: Arbeitsgruppe, Dozenten mit Profilen, Voraussetzungen."""

import streamlit as st

from utils.theming import kapitel_kopf, merkkasten

kapitel_kopf(
    "ℹ️",
    "Über uns",
    "Künstliche Intelligenz und das „Warum“, Sommerakademie Leysin 2026",
)


st.markdown("## Die Arbeitsgruppe")
st.markdown(
    """
Die Arbeitsgruppe **„Eine Einführung in Maschinelles Lernen und Kausalität“**
ist Teil der Sommerakademie Leysin der Studienstiftung des deutschen Volkes.
Wir verbinden Statistik, Informatik und Ökonomie zu einer Frage, die
datengetriebene Forschung überall beschäftigt: **Wann dürfen wir aus Daten auf
Ursache und Wirkung schließen?**

KI-Systeme, die Korrelationen mit kausalen Zusammenhängen verwechseln, können
zu schlechten, oder sogar schädlichen Entscheidungen führen. Wir untersuchen,
wie solche Verzerrungen entstehen, wie man sie erkennt und mit welchen
Strategien man kausale Parameter aus Daten schätzt.
"""
)

st.markdown("## Dozenten")

with st.container(border=True):
    spalte_foto_os, spalte_text_os = st.columns([1, 3])
    with spalte_foto_os:
        st.image(
            "https://oliverschacht.github.io/assets/img/avatar.png",
            use_container_width=True,
        )
    with spalte_text_os:
        st.markdown("### Dr. Oliver Schacht")
        st.markdown(
            """
Oliver hat am Lehrstuhl für Statistik der Universität Hamburg zu 
**Causal Machine Learning** promoviert. Er ist beteiligt am Python-Paket
**DoubleML**, hat zuvor als Data Scientist bei Economic AI gearbeitet und bringt 
über fünf Jahre Lehrerfahrung in Mathematik, Statistik und Data Science mit.
"""
        )
        st.markdown(
            "[Website](https://oliverschacht.github.io/) · "
            "[GitHub](https://github.com/OliverSchacht/) · "
            "[LinkedIn](https://www.linkedin.com/in/oliver-schacht-b5b901200/)"
        )

with st.container(border=True):
    spalte_foto_jtk, spalte_text_jtk = st.columns([1, 3])
    with spalte_foto_jtk:
        st.image(
            "https://janteichertkluge.github.io/content/jan-teichert-kluge_sqr.jpg",
            use_container_width=True,
        )
    with spalte_text_jtk:
        st.markdown("### Jan Teichert-Kluge")
        st.markdown(
            """
Jan promoviert am Lehrstuhl für Statistik der Universität Hamburg zu
**Causal Machine Learning**. Er ist ebenfalls am **DoubleML** Paket beteiligt, 
zuvor hat er Wirtschaftsingenieurwesen studiert und 
mehr als fünf Jahre mit Daten in der Praxis gearbeitet.
"""
        )
        st.markdown(
            "[Website](https://janteichertkluge.github.io/) · "
            "[GitHub](https://github.com/JanTeichertKluge) · "
            "[LinkedIn](https://www.linkedin.com/in/jan-teichert-kluge/)"
        )


merkkasten(
    "Offizielle Programminfo",
    'Details zur Akademie gibt es auf der Seite der '
    '<a href="https://www.studienstiftung.de/kalender/programmlinien/detail/26011102" '
    'target="_blank">Studienstiftung des deutschen Volkes</a>.',
    typ="beispiel",
)

