"""Themen der Gruppenarbeiten: 9 Themen mit Abstracts, offenen Fragen und
Vorbereitungs-Links."""

import streamlit as st

from utils.theming import kapitel_kopf, merkkasten

kapitel_kopf(
    "🧭",
    "Themen der Gruppenarbeiten",
    "Neun Richtungen, die Kapitel dazu und die Fragen, die offen bleiben",
)

st.markdown(
    """
Während der Akademie bearbeitet ihr in Gruppen eines dieser Themen in Python.
Zu jedem Thema findest du hier drei Angaben: den **Abstract**, die **Kapitel**
dieser Website, die dich vorbereiten, und die **offenen Fragen**, also das,
was die Kapitel bewusst nicht beantworten und wo eure Arbeit anfängt.

Die Kapitel liefern das Fundament und je eine Demo. Alles Weitere steht dort
in aufklappbaren Vertiefungen oder gar nicht: Es ist euer Thema, nicht unseres.
"""
)

THEMEN = [
    {
        "emoji": "🏥",
        "titel": "Randomisierte Experimente / Medizin",
        "abstract": (
            "Design und Analyse randomisierter kontrollierter Studien (RCTs) "
            "mit Fokus auf den klinischen Kontext, den methodischen Umgang "
            "mit Non-Compliance und die Auswertung experimenteller Daten in "
            "der medizinischen Forschung."
        ),
        "fragen": [
            "Wie plant man eine Studie, bevor Daten existieren: Fallzahl, Präregistrierung, Stratifizierung?",
            "Non-Compliance und Drop-out: Welche Größe legt man einer Zulassungsbehörde vor, ITT oder LATE?",
            "Zwanzig Endpunkte, einer davon signifikant: Wie geht man mit multiplem Testen um?",
        ],
        "startpunkt": "CONSORT-Statement, `statsmodels.stats.power`, ein echtes Protokoll von ClinicalTrials.gov",
        "links": [
            ("views/kausalitaet/potential_outcomes.py", "Potential Outcomes & RCTs", "⚖️"),
            ("views/kausalitaet/korrelation.py", "Korrelation ≠ Kausalität", "🔀"),
        ],
    },
    {
        "emoji": "🔍",
        "titel": "Explainable ML / AI",
        "abstract": (
            "Methoden zur Interpretation komplexer „Black-Box“-Modelle "
            "(z. B. SHAP, LIME) und die Analyse des grundlegenden Trade-offs "
            "zwischen prädiktiver Performanz und algorithmischer Transparenz."
        ),
        "fragen": [
            "Warum sind SHAP-Werte keine kausalen Effekte, und wo genau bricht die Interpretation bei korrelierten Merkmalen?",
            "Zwei gleich gute Modelle, zwei verschiedene Erklärungen. Welcher soll man glauben?",
            "Was will eine Person eigentlich wissen, deren Kreditantrag abgelehnt wurde?",
        ],
        "startpunkt": "`shap`, `dice-ml`, C. Molnar *Interpretable Machine Learning* (frei online)",
        "links": [
            ("views/ml/explainable_ml.py", "Explainable ML", "🔍"),
            ("views/ml/baeume_ensembles.py", "Trees & Ensembles", "🌲"),
        ],
    },
    {
        "emoji": "💬",
        "titel": "Causality and LLMs",
        "abstract": (
            "Die Schnittstelle von generativer KI und Kausalität: "
            "Einsatzmöglichkeiten von LLMs für Causal Reasoning, Causal "
            "Discovery oder die Kontrolle von Confounding in unstrukturierten "
            "Textdaten."
        ),
        "fragen": [
            "Kann ein LLM einen brauchbaren DAG vorschlagen, und wie misst man das gegen Expertenwissen?",
            "Auswendiglernen oder Schlussfolgern? Was passiert, wenn ihr Namen und Zahlen austauscht?",
            "Text als Confounder: Wie bringt man Arztbriefe oder Bewertungen in eine Kausalanalyse?",
        ],
        "startpunkt": "Benchmarks `Corr2Cause` und `CLADDER`, dazu ein LLM über API",
        "links": [
            ("views/ml/llms_kausalitaet.py", "LLMs & kausales Denken", "💬"),
            ("views/kausalitaet/dags_confounding.py", "DAGs & Confounding", "🕸️"),
        ],
    },
    {
        "emoji": "🕸️",
        "titel": "DAGs / Causal Discovery",
        "abstract": (
            "Grafische Modelle (Directed Acyclic Graphs) und algorithmische "
            "Verfahren zur datengetriebenen Identifikation zugrundeliegender "
            "kausaler Strukturen und valider Kausalpfade direkt aus "
            "Beobachtungsdaten."
        ),
        "fragen": [
            "Das Kapitel setzt den DAG voraus. Lässt er sich aus den Daten selbst lernen (PC, GES, NOTEARS)?",
            "Warum legen Daten den DAG nie vollständig fest? Stichwort Markov-Äquivalenzklasse.",
            "Was bleibt übrig, wenn ein wichtiger Confounder nie erhoben wurde?",
        ],
        "startpunkt": "`causal-learn`, `dowhy`. Zuerst an simulierten Daten mit bekanntem DAG testen",
        "links": [
            ("views/kausalitaet/dags_confounding.py", "DAGs & Confounding", "🕸️"),
            ("views/kausalitaet/korrelation.py", "Korrelation ≠ Kausalität", "🔀"),
        ],
    },
    {
        "emoji": "🎯",
        "titel": "Causal Inference with ML",
        "abstract": (
            "Nutzung flexibler Machine-Learning-Algorithmen zur Schätzung "
            "kausaler Effekte in hochdimensionalen Settings, einschließlich "
            "Frameworks wie Double/Debiased Machine Learning."
        ),
        "fragen": [
            "Für wen wirkt eine Maßnahme? Heterogene Effekte mit Causal Forests und DR-Learner schätzen.",
            "Wie gut müssen die ML-Nuisance-Modelle sein, und wie prüft man das an echten Daten?",
            "Wie stark müsste ein unbeobachteter Confounder sein, um euer Ergebnis zu kippen?",
        ],
        "startpunkt": "`DoubleML` (mitentwickelt in Hamburg), `econml`. Dazu Chernozhukov et al., *Applied Causal Inference Powered by ML and AI*",
        "links": [
            ("views/kausalitaet/kausales_ml.py", "Kausales Machine Learning", "🎯"),
            ("views/ml/baeume_ensembles.py", "Trees & Ensembles", "🌲"),
        ],
    },
    {
        "emoji": "📐",
        "titel": "Tools: RDD / DiD",
        "abstract": (
            "Quasi-experimentelle ökonometrische Designs wie "
            "Difference-in-Differences und Regression Discontinuity Design "
            "zur Schätzung valider kausaler Effekte aus Paneldaten oder bei "
            "exogenen Schwellenwerten (Policy Cutoffs)."
        ),
        "fragen": [
            "Was passiert, wenn Regionen zu unterschiedlichen Zeitpunkten behandelt werden (staggered DiD)?",
            "Wie prüft man Pre-Trends, ohne sich selbst zu betrügen?",
            "Wie wählt man die RDD-Bandbreite datengetrieben, und wie testet man auf Manipulation am Cutoff?",
        ],
        "startpunkt": "`differences`, `linearmodels`, `rdrobust`. Am besten ein echter Politikwechsel mit Paneldaten",
        "links": [
            ("views/kausalitaet/quasi_experimente.py", "Quasi-Experimente: DiD & RDD", "📐"),
            ("views/kausalitaet/potential_outcomes.py", "Potential Outcomes & RCTs", "⚖️"),
        ],
    },
    {
        "emoji": "🎲",
        "titel": "Bayesian Methods",
        "abstract": (
            "Probabilistische Inferenzansätze zur Integration von Vorwissen "
            "(Priors), zur Aktualisierung von Wahrscheinlichkeiten über "
            "Posteriori-Verteilungen und zur Anwendung bayesianischer "
            "Konzepte auf kausale Modellierungen."
        ),
        "fragen": [
            "Wie sehr hängt euer Ergebnis am Prior, und wie zeigt man das ehrlich?",
            "Was tun, wenn keine konjugierte Lösung existiert? MCMC und seine Konvergenzdiagnostik.",
            "Hierarchische Modelle: zwanzig Krankenhäuser, wenige Fälle je Haus. Wie stark soll man poolen?",
        ],
        "startpunkt": "`pymc`, `arviz`. Dazu R. McElreath, *Statistical Rethinking* (Vorlesungsvideos frei)",
        "links": [
            ("views/kausalitaet/bayes.py", "Bayesian Methods", "🎲"),
            ("views/kausalitaet/potential_outcomes.py", "Potential Outcomes & RCTs", "⚖️"),
        ],
    },
    {
        "emoji": "📚",
        "titel": "Examples / Case Studies",
        "abstract": (
            "Angewandte, praxisnahe Replikation und kritische methodische "
            "Analyse prominenter Paper im Bereich Causal Inference sowie die "
            "Übertragung theoretischer Ansätze auf reale Datensätze."
        ),
        "fragen": [
            "Sucht euch ein bekanntes Paper mit offenem Replikationspaket: Kommen dieselben Zahlen heraus?",
            "Welche Annahme trägt das Ergebnis, und wie robust ist es, wenn ihr sie variiert?",
            "Wo weicht das, was das Paper behauptet, von dem ab, was die Daten hergeben?",
        ],
        "startpunkt": "Harvard Dataverse, AEA Data & Code Repository, dazu das Beispielprojekt als Zielstruktur",
        "links": [
            ("views/kausalitaet/korrelation.py", "Korrelation ≠ Kausalität", "🔀"),
            ("views/kausalitaet/quasi_experimente.py", "Quasi-Experimente: DiD & RDD", "📐"),
            ("views/kausalitaet/potential_outcomes.py", "Potential Outcomes & RCTs", "⚖️"),
        ],
    },
    {
        "emoji": "📝",
        "titel": "SEMs / Eigene Experimente & Surveys",
        "abstract": (
            "Formulierung von Strukturgleichungsmodellen (SEMs) zur "
            "Evaluierung komplexer multivariater Zusammenhänge, kombiniert "
            "mit dem praktischen Design, der Implementierung und der "
            "Auswertung eigener Survey-Experimente."
        ),
        "fragen": [
            "Wie misst man Vertrauen, Motivation oder Einstellung, also etwas nicht direkt Beobachtbares?",
            "Wann ist ein SEM überhaupt identifiziert, und woran erkennt man es vorher?",
            "Baut ein eigenes Survey-Experiment: Vignetten, Conjoint, Attention Checks.",
        ],
        "startpunkt": "`semopy` oder `lavaan`, ein Fragebogen-Tool, und vorab eine Power-Analyse",
        "links": [
            ("views/kausalitaet/sem_surveys.py", "SEMs & Survey Experiments", "📋"),
            ("views/kausalitaet/dags_confounding.py", "DAGs & Confounding", "🕸️"),
        ],
    },
]

for zeile_start in range(0, len(THEMEN), 2):
    spalten = st.columns(2)
    for spalte, thema in zip(spalten, THEMEN[zeile_start : zeile_start + 2]):
        with spalte, st.container(border=True):
            st.markdown(f"### {thema['emoji']} {thema['titel']}")
            st.markdown(thema["abstract"])

            st.markdown("**Offene Fragen:**")
            st.markdown("\n".join(f"- {frage}" for frage in thema["fragen"]))
            st.caption(f"Startpunkt: {thema['startpunkt']}")

            st.markdown("**Zur Vorbereitung:**")
            for pfad, label, icon in thema["links"]:
                st.page_link(pfad, label=label, icon=icon)

merkkasten(
    "Und dann?",
    "Sobald eure Gruppe loslegt, bekommt ihr eine eigene Seite auf dieser "
    "Website. Wie das geht, steht in der Projektübersicht. Dort liegt auch "
    "das Beispielprojekt, das den erwarteten Aufbau einer Projektarbeit "
    "vollständig durchspielt.",
    typ="definition",
)

st.page_link("views/projekte/uebersicht.py", label="Zur Projektübersicht", icon="🗂️")
