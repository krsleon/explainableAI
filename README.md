# KI und das „Warum“ | Sommerakademie Leysin 2026

Begleitwebsite zur Arbeitsgruppe **„Eine Einführung in Maschinelles Lernen und
Kausalität“** der Sommerakademie Leysin der Studienstiftung des deutschen
Volkes (18.–27. August 2026, Leysin, Schweiz).

Die Website enthält interaktive Kapitel zu Maschinellem Lernen und Kausalität
sowie die Projektseiten der Gruppen und bleibt nach der Akademie dauerhaft
öffentlich.

**Dozenten:** Dr. Oliver Schacht & Jan Teichert-Kluge (Universität Hamburg,
Lehrstuhl für Statistik)

## Lokal starten

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; unter macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Struktur

```
streamlit_app.py       Einstieg: Navigation, Theme, Projekt-Discovery
views/                 Alle Seiten (Start, Über, Kapitel, Projektübersicht)
  ml/                  Kapitel Maschinelles Lernen
  kausalitaet/         Kapitel Kausalität
content/projekte/      Gruppenprojekte (ein Ordner pro Gruppe)
utils/                 Theming (Farben, Plotly-Template, UI-Bausteine)
                       und Projekt-Discovery
assets/styles.css      Zentrales Stylesheet
tests/                 Smoke-Tests (streamlit.testing AppTest)
```

## Ein Gruppenprojekt hinzufügen

Zwei Wege, die beide mit einem eigenen Ordner unter `content/projekte/` beginnen
(Kleinbuchstaben, Bindestriche, z. B. `gletscher-gang/`):

**Weg 1: Markdown (Standard, kein Streamlit-Wissen nötig).**
`content/projekte/_vorlage/` kopieren, `projekt.md` ausfüllen (Frontmatter:
`titel`, `emoji`, `mitglieder`, `kurzbeschreibung`), Bilder mit in den Ordner
legen und per `![Alt-Text](bild.png)` einbinden. Die Seite erscheint
automatisch in der Navigation.

**Weg 2: eigene Streamlit-Seite.**
Zusätzlich eine `app.py` in den Ordner legen; sie wird statt des Markdowns als
Seite angezeigt. Regeln: kein `st.set_page_config()`, nur Pakete aus
`requirements.txt`, Dateien relativ zum eigenen Ordner laden
(`Path(__file__).parent / "daten.csv"`).

Zwei Vorlagen stehen bereit: `content/projekte/_vorlage/app.py` als minimales
Widget-Skelett und `content/projekte/beispielprojekt/` als ausgearbeitete
Studie an echten Experimentaldaten (Project STAR). Deren Tab-Struktur (Frage,
Daten, naive Analyse, Identifikation, Ergebnis, Limitationen) ist das
empfohlene Gerüst für eine Projektarbeit.

Fehler in einem Projektordner stoppen nie die ganze App; betroffene Seiten
zeigen eine Fehlermeldung.

## Neues Kapitel hinzufügen

1. Datei unter `views/ml/` bzw. `views/kausalitaet/` anlegen (Stubs zeigen das
   Grundgerüst; `utils.theming` liefert `kapitel_kopf`, `merkkasten`,
   `vertiefung` und `gruppen_aufgabe`).
2. In `streamlit_app.py` als `st.Page` in der passenden Sektion registrieren.
3. Seite in `tests/test_smoke.py` zur Liste `STATISCHE_SEITEN` hinzufügen.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```
