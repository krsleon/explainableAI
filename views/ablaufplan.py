"""Ablaufplan der Arbeitsgruppe."""

import streamlit as st

from utils.theming import kapitel_kopf, merkkasten

kapitel_kopf(
    "🗓️",
    "Ablaufplan",
    "Zeitplan der Arbeitsgruppe während der Sommerakademie Leysin 2026",
)

merkkasten(
    "Hinweis",
    "Der Ablauf ist vorläufig und kann sich dynamisch an die Arbeitsgruppe anpassen.",
    typ="achtung",
)

st.markdown("## Ablaufplan")

ablauf = [
    {"Datum": "19.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Akademieeröffnung im Plenum"},
    {
        "Datum": "",
        "Zeit": "10:15-12:30",
        "Programmpunkt": "Arbeitsgruppeneröffnung und Vorstellungsrunde",
    },
    {"Datum": "", "Zeit": "", "Programmpunkt": "Setup: uv, VS Code und Streamlit"},
    {"Datum": "20.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Input: Einführung in ML"},
    {
        "Datum": "",
        "Zeit": "10:15-12:30",
        "Programmpunkt": "Input: Einführung in Kausale Inferenz",
    },
    {"Datum": "21.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Auftakt Gruppenarbeit"},
    {"Datum": "", "Zeit": "10:15-12:30", "Programmpunkt": "Gruppenarbeit"},
    {"Datum": "22.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Gruppenarbeit"},
    {"Datum": "", "Zeit": "10:15-11:30", "Programmpunkt": "Gruppenarbeit"},
    {"Datum": "", "Zeit": "11:30-12:30", "Programmpunkt": "Präsentation Gruppe RCT"},
    {"Datum": "23.08.", "Zeit": "", "Programmpunkt": "frei"},
    {"Datum": "24.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Präsentation Gruppe DAGs"},
    {
        "Datum": "",
        "Zeit": "10:15-11:15",
        "Programmpunkt": "Präsentation Gruppe RDD - DiD",
    },
    {"Datum": "", "Zeit": "11:30-12:30", "Programmpunkt": "Präsentation Gruppe xAI"},
    {"Datum": "25.08.", "Zeit": "09:00-10:00", "Programmpunkt": "Präsentation Gruppe CML"},
    {"Datum": "", "Zeit": "10:15-11:15", "Programmpunkt": "Präsentation Gruppe Bayes"},
    {"Datum": "", "Zeit": "11:30-12:30", "Programmpunkt": "Präsentation Gruppe LLM x Causality"},
    {
        "Datum": "26.08.",
        "Zeit": "09:00-10:00",
        "Programmpunkt": "Fazit, offene Fragen, Diskussion",
    },
    {"Datum": "", "Zeit": "10:15-10:30", "Programmpunkt": "Feedback"},
    {"Datum": "", "Zeit": "10:30-12:30", "Programmpunkt": "Vorbereitung bunter Abend"},
]

zeilen = []
for eintrag in ablauf:
    klassen = []
    if eintrag["Datum"]:
        klassen.append("neuer-tag")
    if not eintrag["Zeit"]:
        klassen.append("ohne-zeit")
    if eintrag["Programmpunkt"] == "frei":
        klassen.append("frei")

    zeilen.append(
        "<tr class=\"{klassen}\">"
        "<td>{datum}</td><td>{zeit}</td><td>{programmpunkt}</td>"
        "</tr>".format(
            klassen=" ".join(klassen),
            datum=eintrag["Datum"],
            zeit=eintrag["Zeit"],
            programmpunkt=eintrag["Programmpunkt"],
        )
    )

st.markdown(
    """
<div class="ablaufplan-wrap">
  <table class="ablaufplan-tabelle">
    <thead>
      <tr>
        <th>Datum</th>
        <th>Zeit</th>
        <th>Programmpunkt</th>
      </tr>
    </thead>
    <tbody>
      {zeilen}
    </tbody>
  </table>
</div>
""".format(zeilen="\n      ".join(zeilen)),
    unsafe_allow_html=True,
)
