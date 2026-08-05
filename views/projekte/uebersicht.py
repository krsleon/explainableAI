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
"""
)

st.page_link(
    "views/projekte/themen.py",
    label="Welche Themen stehen zur Wahl? Zu den Themen der Gruppenarbeiten",
    icon="🧭",
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

st.markdown("## Wie eine Projektarbeit aufgebaut sein sollte")
st.markdown(
    """
Egal ob Markdown oder Streamlit, inhaltlich erwarten wir dieselben sechs
Abschnitte. Das Beispielprojekt zeigt sie an echten Daten:

| Abschnitt | Was hineingehört |
|---|---|
| **1 Frage** | Was wollt ihr wissen, warum ist es interessant, warum ist es schwer? |
| **2 Daten & EDA** | Woher die Daten, was steckt drin, was fehlt? |
| **3 Naive Analyse** | Der einfachste Vergleich, und wo er in die Irre führt |
| **4 Identifikation** | Warum darf man das kausal lesen? Welche Annahmen? |
| **5 Ergebnis** | Effekt mit Unsicherheit, Heterogenität, auch Nullbefunde |
| **6 Limitationen** | Was eure Analyse *nicht* zeigt |

Der wichtigste Abschnitt ist **4**. Eine Zahl ausrechnen kann jedes
Statistikpaket. Zu begründen, warum diese Zahl einen kausalen Effekt misst,
ist die eigentliche Arbeit.
"""
)

merkkasten(
    "Keine Sorge vor Fehlern",
    "Ein Fehler in eurem Projekt legt nie die ganze Website lahm. Schlimmstenfalls "
    "zeigt eure Seite eine Fehlermeldung, und wir beheben das gemeinsam.",
    typ="merke",
)
