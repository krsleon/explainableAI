"""Potential Outcomes & Identification.

Die Seite definiert kausale Effekte, visualisiert das fundamentale Problem der
Kausalinferenz und zeigt, warum Treatment Assignment über die Identifizierbarkeit
entscheidet.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, kapitel_kopf, merkkasten, gruppen_aufgabe

kapitel_kopf(
    "⚖️",
    "Kausale Effekte & kontrafaktische Welten",
    "Potential Outcomes, das fundamentale Problem und die Idee der Identifikation",
)


# ---------------------------------------------------------------- Definition
st.markdown(
    r"""
Eine kausale Aussage vergleicht zwei mögliche Zustände **derselben Einheit**.
Für ein binäres Treatment $D_i\in\{0,1\}$ definieren wir daher zwei potenzielle
Outcomes:

- $Y_i(1)$: das Outcome von Person $i$, **wenn sie behandelt würde**,
- $Y_i(0)$: das Outcome derselben Person, **wenn sie nicht behandelt würde**.

Der **Individual Treatment Effect** ist

$$
\mathrm{ITE} = Y_i(1)-Y_i(0).
$$

Das tatsächlich beobachtete Outcome ist dagegen nur

$$
Y_i = D_iY_i(1)+(1-D_i)Y_i(0).
$$

Wir sehen also immer nur einen der beiden möglichen Zustände.

Der ITE ist deshalb prinzipiell nicht identifizierbar. 
Dies ist das **fundamentale Problem der Kausalen Inferenz**: 
Kausale Inferenz ist im Kern ein Problem fehlender Daten. 
"""
)

merkkasten(
    "Definition",
    "Kausalität ist kontrafaktisch: Ein Treatment-Effekt vergleicht das Outcome "
    "derselben Einheit unter zwei unterschiedlichen Behandlungszuständen. "
    "Beobachten können wir pro Einheit jedoch nur einen dieser Zustände.",
    typ="definition",
)

# ------------------------------------------------ Demo 1: Oracle table
st.markdown("## Demo: Die Orakle-Tabelle und das fundamentale Problem")
st.markdown(
    """
Stell dir zunächst vor, wir hätten Zugriff auf beide potenzielle Outcomes. Die
Tabelle zeigt einen Gesundheits-Score von 0 bis 100 für acht Patient:innen. Im
"Oracle Mode" kennen wir die kontrafaktische Welt; in realen Daten nicht.
"""
)

ORACLE = pd.DataFrame(
    {
        "Person": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Y(0) ohne Behandlung": [45, 50, 55, 60, 75, 80, 85, 90],
        "Y(1) mit Behandlung": [60, 62, 65, 72, 82, 88, 92, 98],
    }
)
ORACLE["individueller Effekt"] = ORACLE["Y(1) mit Behandlung"] - ORACLE["Y(0) ohne Behandlung"]
ORACLE["behandelt"] = ORACLE["Y(0) ohne Behandlung"] < 70

wahrer_ate = ORACLE["individueller Effekt"].mean()

oracle_modus = st.toggle("Oracle Mode: beide potenziellen Outcomes sichtbar", value=True)

if oracle_modus:
    st.dataframe(ORACLE.drop(columns=["behandelt"]), hide_index=True, use_container_width=True)
    st.metric("Wahrer Average Treatment Effect (ATE)", f"{wahrer_ate:+.1f} Punkte")
    st.caption(
        "Hier können wir den kausalen Effekt tatsächlich berechnen, weil die "
        "Simulation beide Welten kennt."
    )
else:
    beobachtet = ORACLE.copy()
    beobachtet["Treatment"] = np.where(beobachtet["behandelt"], "Behandlung", "Kontrolle")
    beobachtet["beobachtetes Y"] = np.where(
        beobachtet["behandelt"],
        beobachtet["Y(1) mit Behandlung"],
        beobachtet["Y(0) ohne Behandlung"],
    )
    st.dataframe(
        beobachtet[["Person", "Treatment", "beobachtetes Y"]],
        hide_index=True,
        use_container_width=True,
    )
    mit = beobachtet.loc[beobachtet["behandelt"], "beobachtetes Y"].mean()
    ohne = beobachtet.loc[~beobachtet["behandelt"], "beobachtetes Y"].mean()
    naive = mit - ohne
    c1, c2 = st.columns(2)
    c1.metric("Beobachtete Gruppendifferenz", f"{naive:+.1f} Punkte")
    st.error(
        "Mit dem Umschalten ist pro Person genau ein Potential Outcome verschwunden. "
        "Das andere ist kontrafaktisch. Der individuelle Treatment-Effekt ist damit "
        "nicht direkt beobachtbar."
    )

st.markdown(
    r"""
Dieses fehlende Gegenstück ist das **fundamentale Problem der Kausalinferenz**.
Es ist kein Problem, das ein größerer Datensatz oder ein leistungsfähigeres
Machine-Learning-Modell für die einzelne Person einfach lösen kann.

Auf Populationsebene definieren wir deshalb beispielsweise den
**Average Treatment Effect**

$$
\mathrm{ATE}=E[Y(1)-Y(0)] = E[Y(1)]-E[Y(0)].
$$

Auch dieser Estimand enthält zunächst kontrafaktische Größen. Die entscheidende
Frage lautet daher: **Unter welchen Annahmen können wir ihn durch beobachtbare
Größen ausdrücken?**

Wie wir sehen, entscheidet der **Treatment Assignment Mechanismus** darüber, ob die
beobachteten Daten die gewünschte kausale Größe enthalten. In randomisierten
Experimenten ist die Treatment-Zuweisung unabhängig von den potenziellen Outcomes.
In beobachteten Daten ist sie es in der Regel nicht. Die Verzerrung auf Populationsebene
lässt sich hier allgemein zerlegen.

Für den einfachen Mittelwertvergleich (*Simple Difference in Outcomes*, SDO) gilt:

$$
\underbrace{E[Y \mid D{=}1] - E[Y \mid D{=}0]}_{\text{SDO}}
= \mathrm{ATE}
+ \underbrace{E\big[Y^0 \mid D{=}1\big] - E\big[Y^0 \mid D{=}0\big]}_{\text{Selection Bias}}
+ \underbrace{(1 - \pi)\,(\mathrm{ATT} - \mathrm{ATU})}_{\text{Heterogeneous Treatment Effect Bias}},
$$

wobei $\pi$ den Anteil der Behandelten bezeichnet. Der Selection Bias
vergleicht die *kontrafaktischen* unbehandelten Outcomes beider Gruppen. Er
ist genau dann von null verschieden, wenn sich Behandelte und Unbehandelte
auch ohne Behandlung unterschieden hätten, wie in der Tabelle oben, in der
gerade die Kränkeren zum Medikament greifen.

"""
)


# ---------------------------------------- Estimand / Identification / Estimation
st.markdown("## Von der Frage zur Zahl")

begriffe = pd.DataFrame(
    {
        "Schritt": ["1. Kausale Frage", "2. Estimand", "3. Identification", "4. Estimation", "5. Estimate"],
        "Leitfrage": [
            "Welche Intervention interessiert uns?",
            "Welche kausale Zielgröße beschreibt diese Frage?",
            "Unter welchen Annahmen ist die Zielgröße aus beobachtbaren Daten lernbar?",
            "Mit welcher Rechenvorschrift schätzen wir die identifizierte Größe?",
            "Welchen Zahlenwert erhalten wir in dieser Stichprobe?",
        ],
        "Beispiel": [
            "Wirkt die Behandlung?",
            "ATE",
            "z. B. zufällige Zuweisung / geeignete Vergleichbarkeit",
            "z. B. Differenz der Gruppenmittelwerte",
            "+7.4 Punkte",
        ],
    }
)
st.dataframe(begriffe, hide_index=True, use_container_width=True)

merkkasten(
    "Identification vor Estimation",
    "Ein ausgefeilter Estimator kann ein Identifikationsproblem nicht reparieren. "
    "Bevor wir and Algorithmen, Standardfehler oder Machine Learning denken, "
    "müssen wir begründen, warum die beobachteten Daten überhaupt die gewünschte "
    "kausale Größe enthalten.",
    typ="merke",
)


merkkasten(
    "Begriffe: Estimand, Estimator, Estimate",
    "Drei Begriffe, die in der gesamten Kausalinferenz sauber getrennt "
    "werden: Der <b>Estimand</b> ist die Zielgröße, die wir wissen wollen "
    "(etwa der ATE), definiert über potenzielle Outcomes und damit nicht "
    "direkt beobachtbar. Ein <b>Estimator</b> ist die Rechenvorschrift, die "
    "aus Daten eine Antwort produziert (etwa die Differenz der "
    "Gruppenmittelwerte). Das <b>Estimate</b> ist der konkrete Zahlenwert, "
    "den der Estimator in einer Stichprobe liefert. <b>Identification</b> "
    "bezeichnet den Schritt, den Estimand unter Annahmen durch beobachtbare "
    "Größen auszudrücken, erst danach beginnt die <b>Estimation</b>. Ein "
    "präziser Estimator nützt nichts, wenn die Identification scheitert.",
    typ="definition",
)


# ------------------------------------------ Demo 2: Assignment mechanism
st.markdown("## Demo: Der Zuteilungsmechanismus entscheidet")
st.markdown(
    r"""
Warum kann der einfache Vergleich von Treatment- und Kontrollgruppe manchmal
einen kausalen Effekt identifizieren und manchmal nicht? Entscheidend ist, **wie
Menschen in die Behandlung gelangen**.

Wir simulieren einen Gesundheits-Score mit einem wahren Treatment-Effekt von
$+8$ Punkten. Krankheitsschwere beeinflusst das Outcome. In Beobachtungsdaten
kann sie zusätzlich beeinflussen, wer behandelt wird.
"""
)

zuteilung = st.radio(
    "Treatment Assignment",
    ["Selbstselektion / ärztliche Auswahl", "Randomisierte Zuweisung"],
    horizontal=True,
)
n_studie = st.select_slider(
    "Stichprobengröße",
    options=[50, 100, 200, 500, 1000, 5000],
    value=200,
)

WAHRER_EFFEKT = 8.0


@st.cache_data
def studie_simulieren(randomisiert: bool, n: int, seed: int = 4):
    rng = np.random.default_rng(seed)
    schwere = rng.uniform(0, 1, n)
    if randomisiert:
        d = rng.integers(0, 2, n)
    else:
        # Schwerere Patient:innen werden mit höherer Wahrscheinlichkeit behandelt.
        d = (rng.uniform(0, 1, n) < 0.10 + 0.80 * schwere).astype(int)
    y0 = 85 - 30 * schwere + rng.normal(0, 5, n)
    y = y0 + WAHRER_EFFEKT * d
    return schwere, d, y


schwere, d, y_beob = studie_simulieren(zuteilung.startswith("Randomisierte"), n_studie)

diff = y_beob[d == 1].mean() - y_beob[d == 0].mean()
se = np.sqrt(
    y_beob[d == 1].var(ddof=1) / max((d == 1).sum(), 1)
    + y_beob[d == 0].var(ddof=1) / max((d == 0).sum(), 1)
)

spalte_balance, spalte_schaetzung = st.columns(2)
with spalte_balance:
    fig_balance = go.Figure()
    fig_balance.add_bar(
        x=["Kontrolle", "Treatment"],
        y=[schwere[d == 0].mean(), schwere[d == 1].mean()],
        marker_color=[FARBEN["gletscher"], FARBEN["sonne"]],
    )
    fig_balance.update_layout(
        title="Sind die Gruppen vergleichbar?",
        yaxis_title="Ø Krankheitsschwere",
        yaxis={"range": [0, 1]},
        height=360,
    )
    st.plotly_chart(fig_balance, use_container_width=True)

with spalte_schaetzung:
    fig_schaetz = go.Figure()

    fig_schaetz.add_scatter(
        x=["Gruppendifferenz"],
        y=[diff],
        mode="markers",
        marker={"color": FARBEN["nacht"], "size": 12,},
        error_y={
            "type": "data",
            "array": [1.96 * se],
            "visible": True,
        },
        name="Geschätzter Effekt",
    )

    fig_schaetz.add_hline(
        y=WAHRER_EFFEKT,
        line_dash="dash",
        line_color=FARBEN["wiese"],
        annotation_text="wahrer ATE (+8)",
    )

    fig_schaetz.update_layout(
        title="Geschätzter Effekt (± 95%-Intervall)",
        yaxis_title="Punkte",
        height=360,
        showlegend=False,
    )

    st.plotly_chart(fig_schaetz, use_container_width=True)

if zuteilung.startswith("Randomisierte"):
    st.success(
        "**Balance:** Beide Gruppen sind im Mittel gleich krank, der "
        "Gruppenvergleich isoliert also den Medikamenteneffekt. Mit "
        "wachsender Studiengröße zieht sich das Intervall um den wahren "
        "Wert zusammen."
    )
else:
    st.error(
        "**Unbalanciert:** Die Medikamentengruppe ist deutlich kränker. Der "
        "naive Vergleich vermengt Medikamentenwirkung und Krankheitsschwere, "
        "und mehr Daten helfen hier nicht: Der Bias bleibt bestehen, nur das "
        "Intervall wird enger. Man schätzt präzise das Falsche."
    )

st.markdown(
    r"""
Randomisierung ist damit **eine** besonders transparente Identifikationsstrategie:

$$
(Y(1),Y(0)) \perp D.
$$

Die Zuteilung enthält keine Information über die potenziellen Outcomes,
alle Mechanismen der Selbstselektion sind ausgeschaltet. Daraus folgt
$E\big[Y^0 \mid D{=}1\big] = E\big[Y^0 \mid D{=}0\big]$: Selection Bias und
Heterogenitätsbias verschwinden, und der einfache Mittelwertvergleich
identifiziert den ATE:

$$
E[Y \mid D{=}1] - E[Y \mid D{=}0] = E\big[Y^1\big] - E\big[Y^0\big] = \mathrm{ATE}.
$$

Randomisierung ist aber nicht die einzige Identifikationsstrategie. Viele Forschungsfragen lassen keine
Randomisierung zu. Dann müssen andere Designs oder Annahmen begründen, warum die
beobachteten Vergleichsgruppen ein glaubwürdiges Counterfactual liefern.
"""
)

merkkasten(
    "Für die weitere AG",
    "RCTs, DAGs, natürliche Experimente und Causal ML unterscheiden sich in ihren "
    "Werkzeugen. Gemeinsam ist ihnen die Aufgabe, aus beobachteten Daten unter "
    "expliziten Annahmen ein glaubwürdiges kontrafaktisches Vergleichsszenario "
    "zu konstruieren.",
    typ="merke",
)

gruppen_aufgabe(
    "Ideen für die Gruppenarbeit:",
    [
        (
            "Wie plant man ein Experiment, <i>bevor</i> Daten existieren? "
            "Fallzahlberechnung, Präregistrierung, Block- oder "
            "Stratifizierungs-Randomisierung. Hier entscheidet sich, ob eine "
            "Studie überhaupt etwas zeigen kann."
        ),
        (
            "Was tun, wenn nicht alle mitmachen? "
            "(ITT, Per-Protocol und IV) "
        ),
        (
            "Wie geht man mit mehreren Endpunkten um? Wer zwanzig Outcomes "
            "testet, findet fast garantiert eines mit p &lt; 0,05. Bonferroni, "
            "Holm, False Discovery Rate: Was davon ist in klinischen Studien "
            "Standard?"
        ),
    ],
    hinweis=(
        "Startpunkt: Das CONSORT-Statement als Checkliste für "
        "RCT-Berichte, <code>statsmodels.stats.power</code> für "
        "Fallzahlen."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- S. Cunningham (2021), *Causal Inference: The Mixtape*, Yale University Press, Kap. 4 (frei online)
- M. Huber (2023), *Causal Analysis: Impact Evaluation and Causal Machine Learning with Applications in R*, MIT Press, Kap. 1 bis 3
- G. W. Imbens & D. B. Rubin (2015), *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*, Cambridge University Press
"""
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Nächster Schritt: Welche Annahmen tragen die Identifikation?")
st.markdown(
    """
Wir haben gesehen, dass der Treatment-Assignment-Mechanismus entscheidend ist.
In Beobachtungsdaten ist er selten zufällig. Kausale Modelle helfen dann, die
notwendigen Annahmen explizit zu machen und zu prüfen, **welche Variablen einen
kausalen Vergleich ermöglichen**.
"""
)

st.page_link(
    "views/kausalitaet/dags_confounding.py",
    label="Weiter: Identifikation, Annahmen & Landkarte der AG",
    icon="🧭",
)
