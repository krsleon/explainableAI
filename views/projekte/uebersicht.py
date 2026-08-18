"""Projektgalerie: alle Gruppenprojekte + Anleitung zum Mitmachen."""

import streamlit as st

from utils.projects import lade_projekte
from utils.theming import kapitel_kopf, merkkasten

kapitel_kopf(
    "🗂️",
    "Gruppenprojekte",
    "Eigene Themen, erkundet in Python, von den Gruppen der Akademie",
)

st.markdown(
    """
Während der Akademie arbeitet ihr in Gruppen an eigenen Fragestellungen rund
um Maschinelles Lernen und Kausalität. Jede Gruppe bekommt hier eine eigene
Seite, die Galerie unten füllt sich im Lauf der Akademie.

Aktuell befindet sich hier ein Beispielprojekt von den Dozenten.
"""
)


projekte = lade_projekte()
seiten_map = st.session_state.get("projekt_seiten", {})

if not projekte:
    st.info("Noch keine Projekte eingereicht. Die Galerie füllt sich während der Akademie.")
else:
    for zeile_start in range(0, len(projekte), 3):
        spalten = st.columns(3)
        for spalte, projekt in zip(spalten, projekte[zeile_start : zeile_start + 3]):
            with spalte, st.container(border=True):
                st.markdown(f"### {projekt.emoji} {projekt.titel}")
                if projekt.kurzbeschreibung:
                    st.markdown(projekt.kurzbeschreibung)
                if projekt.mitglieder:
                    st.caption("Team: " + ", ".join(projekt.mitglieder))
                if projekt.fehler:
                    st.warning(projekt.fehler)
                seite = seiten_map.get(projekt.slug)
                if seite is not None:
                    st.page_link(seite, label="Zur Projektseite", icon="▶️")

st.markdown("## So kommt euer Projekt auf die Website")

with st.expander("Weg 1 (Standard): Markdown-Datei, kein Streamlit-Wissen nötig"):
    st.markdown(
        """
1. Kopiert den Ordner `content/projekte/_vorlage/` und benennt ihn nach eurem
   Projekt, z. B. `content/projekte/gletscher-gang/` (Kleinbuchstaben,
   Bindestriche statt Leerzeichen).
2. Füllt die `projekt.md` aus: oben im **Frontmatter** stehen Titel, Emoji,
   Teammitglieder und eine Kurzbeschreibung, darunter schreibt ihr ganz
   normales Markdown.
3. Bilder (Plots, Screenshots) legt ihr einfach mit in euren Ordner und bindet
   sie mit `![Beschreibung](mein-plot.png)` ein.
4. Änderungen als Pull Request einreichen. Nach dem Merge erscheint eure
   Seite automatisch in der Navigation.
"""
    )

with st.expander("Weg 2 (Kür): eigene interaktive Streamlit-Seite"):
    st.markdown(
        """
Ihr wollt Slider, Live-Plots und eigene Widgets? Legt **zusätzlich** zur
`projekt.md` eine `app.py` in euren Ordner, dann wird diese als Seite
angezeigt (die `projekt.md` liefert weiterhin Titel & Infos für diese
Galerie).

Zwei Vorlagen stehen bereit:

- `content/projekte/_vorlage/app.py`: minimales Skelett aus Schieberegler,
  Berechnung und Plot. Gut zum Loslegen.
- `content/projekte/beispielprojekt/`: eine vollständig ausgearbeitete
  Studie an echten Experimentaldaten. Zeigt, wie eine Projektarbeit am Ende
  aussehen kann.

**Spielregeln für `app.py`:**

- Kein `st.set_page_config()` aufrufen, das erledigt die Haupt-App bereits.
- Nur Pakete aus `requirements.txt` verwenden (numpy, pandas, plotly,
  scikit-learn, scipy, statsmodels). Braucht ihr mehr, sprecht uns an.
- Dateien (Daten, Bilder) relativ zu eurem Ordner laden, z. B. mit
  `Path(__file__).parent / "daten.csv"`.
- Widget-`key`s mit eurem Projektnamen präfixen, damit sie sich nicht mit
  anderen Seiten überschneiden.
"""
    )

st.markdown("## Was soll am Ende entstehen?")

st.markdown(
    """
Ziel ist **nicht**, euer gesamtes Themengebiet abzudecken. Wählt innerhalb
eures Themas eine **präzise wissenschaftliche Frage** und untersucht sie anhand
eines geeigneten Beispiels, Datensatzes, Experiments oder einer Simulation.

Eure Streamlit-App soll dabei nicht bloß Ergebnisse präsentieren. Sie soll eine
zentrale methodische Idee, Annahme oder Grenze eures Themas **interaktiv
untersuchbar** machen.
"""
)

st.markdown("### Eine sinnvolle Struktur")

st.markdown(
    """
| Abschnitt | Leitfrage |
|---|---|
| **1 Frage & Motivation** | Was genau wollt ihr wissen? Warum ist die Frage interessant und nicht trivial? |
| **2 Konzept / Methode** | Welche zentrale methodische Idee braucht man, um die Frage zu bearbeiten? |
| **3 Annahmen & Gültigkeit** | Was muss gelten, damit eure Schlussfolgerung trägt? Bei klassischen kausalen Analysen: Was identifiziert euren Effekt? |
| **4 Daten / Experiment / Simulation** | Woran untersucht ihr die Frage und warum eignet sich dieses Setup? |
| **5 Interaktive Analyse** | Was können Nutzer:innen verändern und was lässt sich dadurch erkennen? |
| **6 Findings** | Was habt ihr tatsächlich herausgefunden? |
| **7 Grenzen & offene Fragen** | Was zeigt eure Analyse nicht? Welche Annahme oder Designentscheidung ist besonders kritisch? |
"""
)

merkkasten(
    "Wissenschaftliche Aussage vor technischer Umsetzung",
    "Eine Methode auszuführen oder eine Zahl zu berechnen ist nicht das Hauptziel. "
    "Entscheidend ist, dass ihr begründen könnt, <b>welche wissenschaftliche "
    "Aussage eure Analyse erlaubt – und welche nicht</b>.",
    typ="definition",
)

st.markdown("### Was die Streamlit-App leisten soll")

st.markdown(
    """
Die App sollte mindestens **eine wissenschaftlich relevante Variation**
ermöglichen. Beispielsweise könnt ihr

- eine Annahme oder einen Parameter verändern,
- verschiedene Modelle oder Auswertungen vergleichen,
- eine Datenstruktur oder Stichprobengröße variieren,
- einen Prior, Cutoff oder Prompt verändern,
- oder gezielt einen Fall erzeugen, in dem die Methode scheitert.

**Leitfrage:** Was können andere durch die Interaktion verstehen, was auf einer
statischen Folie weniger gut sichtbar wäre?

Eine kleine, präzise App ist dabei besser als ein großes Dashboard ohne klare
wissenschaftliche Aussage.
"""
)

st.markdown("### Beispielhafter Arbeitsplan für die vier Projektblöcke")

st.markdown(
    """
| Block | Ziel am Ende |
|---|---|
| **1 · Scope & Design** (21.8.) | Forschungsfrage, Konzept, Daten/Simulation, zentrale Annahme und App-Skizze stehen fest. |
| **2 · Analyse** (21.8.) | Die zentrale Analyse funktioniert und eine erste Visualisierung existiert. |
| **3 · Streamlit & Story** (22.8.) | Interaktion, wissenschaftliche Erklärung, Findings und Limitationen sind integriert. |
| **4 · Finalisierung** (22.8.) | App und Präsentation sind fertig; zentrale Aussage und Grenzen wurden getestet. |
"""
)

st.info(
    "**Wichtig:** Ihr müsst nicht alle offenen Fragen eures Themas bearbeiten. "
    "Eine gut untersuchte Frage ist besser als drei nur angerissene."
)

merkkasten(
    "Keine Sorge vor Fehlern",
    "Ein Fehler in eurem Projekt legt nie die ganze Website lahm. Schlimmstenfalls "
    "zeigt eure Seite eine Fehlermeldung, und wir beheben das gemeinsam.",
    typ="merke",
)
