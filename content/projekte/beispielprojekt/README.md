# Beispielprojekt: Project STAR

Vorlage für die Gruppenprojekte der Sommerakademie. Zeigt an echten Daten aus
einem randomisierten Experiment, wie eine vollständige Projektarbeit aufgebaut
sein kann.

**Frage:** Bringen kleinere Schulklassen bessere Leistungen?
**Daten:** Project STAR, Tennessee 1985–1989, Kindergarten-Kohorte (6.325 Kinder)

## Aufbau

```
beispielprojekt/
├── projekt.md      Frontmatter für die Galerie (Titel, Emoji, Team)
├── app.py          Streamlit-Seite: nur Darstellung, keine Berechnung
├── analyse.py      alle Berechnungen, ohne Streamlit-Import
├── README.md       diese Datei
└── daten/
    ├── star_kindergarten.csv
    └── QUELLE.md   Herkunft, Lizenz, Aufbereitung
```

## Die sechs Abschnitte

Die Tabs in `app.py` sind die sechs Abschnitte, die wir auch von euren
Projekten sehen wollen:

| Tab | Was hineingehört |
|-----|------------------|
| 1 Frage | Was wollt ihr wissen, warum ist es interessant, warum ist es schwer? |
| 2 Daten & EDA | Woher die Daten, was steckt drin, was fehlt? |
| 3 Naive Analyse | Der einfachste Vergleich, und wo er in die Irre führt |
| 4 Identifikation | Warum darf man das kausal lesen? Welche Annahmen? |
| 5 Ergebnis | Effekt mit Unsicherheit, Heterogenität, Nullbefunde |
| 6 Limitationen | Was die Analyse *nicht* zeigt |

Der wichtigste Abschnitt ist **4**. Eine Zahl auszurechnen kann jedes
Statistikpaket. Zu begründen, warum diese Zahl einen kausalen Effekt misst,
ist die eigentliche Arbeit.

## Warum Analyse und Darstellung getrennt sind

`analyse.py` importiert kein Streamlit. Das hat drei Gründe:

1. Die Funktionen sind ohne laufende App testbar (siehe
   `tests/test_beispielprojekt.py` im Repo-Wurzelverzeichnis).
2. Ihr könnt dieselben Funktionen in einem Jupyter-Notebook benutzen.
3. Wer den Code liest, findet die Statistik an einer Stelle statt verteilt
   zwischen Widgets.

Für eure Projekte ist das eine Empfehlung, keine Pflicht, aber eine gute.

## Lokal ausführen

Aus dem Wurzelverzeichnis des Repos:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Die Seite erscheint in der Navigation unter „Gruppenprojekte“.

Nur die Analysefunktionen, ohne App:

```python
import sys; sys.path.insert(0, "content/projekte/beispielprojekt")
import analyse

daten = analyse.lade_daten()
print(analyse.ate_mit_ci(daten))
```

## Technische Spielregeln (gelten für alle Projekte)

- Kein `st.set_page_config()`, das macht die Haupt-App.
- Nur Pakete aus `requirements.txt`. Braucht ihr mehr, sprecht uns an.
- Dateien relativ zum eigenen Ordner laden: `Path(__file__).parent / "daten.csv"`.
- Wollt ihr eigene Module importieren, muss der Projektordner in `sys.path`
  (siehe Kopf von `app.py`), denn Streamlit startet die Seite aus dem
  Wurzelverzeichnis.
- Widget-`key`s mit dem Projektnamen präfixen, damit sie sich nicht mit
  anderen Seiten überschneiden.

## Quellen

- A. B. Krueger (1999), *Experimental Estimates of Education Production
  Functions*, Quarterly Journal of Economics 114(2), 497–532.
- C. Kleiber & A. Zeileis (2008), *Applied Econometrics with R*, Springer
  (R-Paket `AER`, Datensatz `STAR`).
