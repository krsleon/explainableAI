"""Kapitel Kausalität: SEMs & Survey Experiments.

Strukturgleichungsmodelle als DAGs mit Gleichungen (Mediationsanalyse
interaktiv) und das Handwerk eigener Survey-Experimente inkl. Power-Rechner.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "📋",
    "SEMs & Survey Experiments",
    "Wirkmechanismen zerlegen und eigene Daten erheben, die Antworten erlauben",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Ein **Strukturgleichungsmodell (SEM)** ist ein DAG, in dem an jedem Pfeil
eine **Gleichung** hängt: Jede Variable wird als Funktion ihrer direkten
Ursachen (plus Störterm) geschrieben. Damit kann man nicht nur fragen *„wirkt
X auf Y?“*, sondern **„auf welchem Weg?“**. Die Königsdisziplin dieser
Fragestellung heißt **Mediation Analysis**.
"""
)

# ------------------------------------------------ Demo 1: Mediation
st.markdown("## Demo: Mediationsanalyse")
st.markdown(
    r"""
Beispiel: Wirkt **Social-Media-Zeit** ($X$) auf das **Wohlbefinden** ($Y$)?
Und falls ja: direkt, oder vermittelt über schlechteren **Schlaf** ($M$)?
Das zugehörige SEM besteht aus zwei Strukturgleichungen:

$$
M = a\,X + \varepsilon_M, \qquad
Y = c'\,X + b\,M + \varepsilon_Y .
$$

Einsetzen der ersten Gleichung in die zweite liefert die **Effektzerlegung**:

$$
Y = (c' + a\,b)\,X + b\,\varepsilon_M + \varepsilon_Y
\qquad\Longrightarrow\qquad
\tau_{\text{total}} = \underbrace{c'}_{\text{direkt}} + \underbrace{a \cdot b}_{\text{indirekt}} .
$$

Geschätzt wird mit drei Regressionen: $Y$ auf $X$ ergibt den totalen Effekt,
$Y$ auf $X$ und $M$ ergibt $c'$ und $b$, und $M$ auf $X$ ergibt $a$. Stelle
die wahren Pfade ein und vergleiche mit den Schätzungen (Simulation,
$n = 2\,000$):
"""
)

regler_a, regler_b, regler_c = st.columns(3)
pfad_a = regler_a.slider("a: Social Media → Schlaf", -1.0, 1.0, -0.6, step=0.1)
pfad_b = regler_b.slider("b: Schlaf → Wohlbefinden", -1.0, 1.0, 0.7, step=0.1)
pfad_c = regler_c.slider("c′: direkter Pfad X → Y", -1.0, 1.0, -0.2, step=0.1)


@st.cache_data
def mediation_daten(a: float, b: float, c: float, n: int = 2000, seed: int = 14):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = a * x + rng.normal(0, 1, n)
    y = c * x + b * m + rng.normal(0, 1, n)
    return x, m, y


x, m, y = mediation_daten(pfad_a, pfad_b, pfad_c)

total_dach = np.polyfit(x, y, 1)[0]
koeffs = np.linalg.lstsq(np.column_stack([x, m, np.ones_like(x)]), y, rcond=None)[0]
direkt_dach, b_dach = koeffs[0], koeffs[1]
a_dach = np.polyfit(x, m, 1)[0]
indirekt_dach = a_dach * b_dach

spalte_dag, spalte_effekte = st.columns([2, 3])
with spalte_dag:
    st.graphviz_chart(
        f"""
digraph {{
    rankdir=LR; bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#EEF3FA",
          color="#3E6DB5", fontname="sans-serif"];
    edge [fontname="sans-serif", color="#5C6470"];
    X [label="📱 Social Media (X)"];
    M [label="😴 Schlaf (M)"];
    Y [label="🙂 Wohlbefinden (Y)"];
    X -> M [label="a = {pfad_a:.1f}"];
    M -> Y [label="b = {pfad_b:.1f}"];
    X -> Y [label="c′ = {pfad_c:.1f}", color="#E8804C", fontcolor="#B05A2B"];
}}
""",
        use_container_width=True,
    )
with spalte_effekte:
    wahr = [pfad_c + pfad_a * pfad_b, pfad_c, pfad_a * pfad_b]
    geschaetzt = [total_dach, direkt_dach, indirekt_dach]
    namen = ["Totaler Effekt", "Direkter Pfad c′", "Indirekter Pfad a·b"]
    fig_med = go.Figure()
    fig_med.add_bar(name="wahr", x=namen, y=wahr, marker_color=FARBEN["schiefer"])
    fig_med.add_bar(name="geschätzt", x=namen, y=geschaetzt, marker_color=FARBEN["gletscher"])
    fig_med.update_layout(barmode="group", yaxis_title="Effekt", height=380)
    st.plotly_chart(fig_med, use_container_width=True)

st.markdown(
    """
Typischer Befund bei den Startwerten: Der totale Effekt ist deutlich negativ,
obwohl der *direkte* Pfad klein ist. Der Wirkmechanismus läuft über den
Schlaf. Solche Zerlegungen sind für die Gestaltung von Interventionen
unmittelbar relevant: Statt der Bildschirmzeit selbst wäre hier der
Schlafkanal der wirksamere Ansatzpunkt.
"""
)

merkkasten(
    "Achtung: Mediation ist kausal anspruchsvoll",
    "Die Zerlegung stimmt nur, wenn das SEM stimmt: <b>kein unbeobachteter "
    "Confounder zwischen M und Y</b> (die klassische Schwachstelle, denn X "
    "mag randomisiert sein, M ist es fast nie), korrekte Pfeilrichtungen, "
    "keine vergessenen Pfade. Ein SEM ist so glaubwürdig wie sein DAG.",
    typ="achtung",
)

st.markdown(
    """
**In der Praxis:** SEMs mit **latenten Variablen** (Konstrukte wie
„Wohlbefinden“, gemessen über mehrere Fragebogen-Items samt Messmodell)
schätzt man mit `lavaan` (R) oder `semopy` (Python). Unsere Demo ist ein
SEM mit direkt beobachteten Variablen, das Prinzip ist jedoch dasselbe.
"""
)

# ------------------------------------------ Survey-Experimente
st.markdown("## Eigene Daten: Survey-Experimente")
st.markdown(
    """
Manchmal existieren die Daten für deine Frage schlicht nicht. Dann erhebst
du sie selbst, und die stärkste Variante dafür ist das
**Survey Experiment**: ein vollwertiger RCT im Fragebogen, denn die
Randomisierung liegt in deiner Hand.

- **Framing Experiment:** Die eine Hälfte liest „Klimaabgabe“, die andere
  „Klimaprämie“. Identische Politik, zufällig zugeteilte Wortwahl. Der
  Unterschied in der Zustimmung ist ein kausaler Framing-Effekt.
- **Vignette Experiment:** Beschreibungen fiktiver Personen oder Szenarien,
  in denen Merkmale wie Geschlecht, Herkunft oder Beruf zufällig variieren.
  Auf diese Weise lässt sich etwa Diskriminierung kausal messen.
- **Handwerksregeln:** eine Manipulation pro Vergleich, ein
  Manipulation Check, die Auswertung vorab festlegen (Preregistration), und
  vor der Erhebung klären, ob die Stichprobe ausreicht. Das führt zur Power.
"""
)

with vertiefung("Power-Analyse für die eigene Stichprobengröße"):
    st.markdown(
        r"""
    Die **Power** ist die Wahrscheinlichkeit, einen tatsächlich vorhandenen
    Effekt der standardisierten Größe $d$ (Cohens $d$) mit einem zweiseitigen
    Test zum Niveau $\alpha = 0{,}05$ zu entdecken. In Normal-Approximation gilt
    für zwei Gruppen der Größe $n$:

    $$
    \mathrm{Power} \approx \Phi\!\Big(d\,\sqrt{\tfrac{n}{2}} \;-\; z_{1-\alpha/2}\Big),
    $$

    wobei $\Phi$ die Verteilungsfunktion der Standardnormalverteilung bezeichnet.
    """
    )

    regler_d, regler_n = st.columns(2)
    effekt_d = regler_d.slider(
        "Erwartete Effektgröße (Cohens d)", 0.1, 1.0, 0.4, step=0.05,
        help="0,2 = klein · 0,5 = mittel · 0,8 = groß",
    )
    n_gruppe = regler_n.slider("Teilnehmende pro Gruppe", 10, 500, 100, step=10)


    def power_zweistichproben(d: float, n: int, alpha: float = 0.05) -> float:
        """Power eines zweiseitigen Zwei-Gruppen-Tests (Normal-Approximation)."""
        z_krit = stats.norm.ppf(1 - alpha / 2)
        verschiebung = d * np.sqrt(n / 2)
        return float(1 - stats.norm.cdf(z_krit - verschiebung) + stats.norm.cdf(-z_krit - verschiebung))


    n_raster = np.arange(10, 501, 5)
    power_kurve = [power_zweistichproben(effekt_d, n) for n in n_raster]
    aktuelle_power = power_zweistichproben(effekt_d, n_gruppe)

    fig_power = go.Figure()
    fig_power.add_scatter(
        x=n_raster, y=power_kurve, mode="lines", name="Power",
        line=dict(color=FARBEN["nacht"], width=3),
    )
    fig_power.add_hline(
        y=0.8, line_dash="dash", line_color=FARBEN["wiese"],
        annotation_text="Konvention: 80 %",
    )
    fig_power.add_scatter(
        x=[n_gruppe], y=[aktuelle_power], mode="markers", name="deine Studie",
        marker=dict(color=FARBEN["sonne"], size=14, symbol="star"),
    )
    fig_power.update_layout(
        xaxis_title="Teilnehmende pro Gruppe",
        yaxis_title="Power (Chance, einen echten Effekt zu finden)",
        yaxis=dict(range=[0, 1.02]), height=400,
    )
    st.plotly_chart(fig_power, use_container_width=True)

    st.metric("Power deiner Konfiguration", f"{aktuelle_power:.0%}")
    if aktuelle_power < 0.8:
        st.warning(
            "Power unter 80 %: Ein tatsächlich vorhandener Effekt dieser Größe "
            "bliebe mit hoher Wahrscheinlichkeit unentdeckt. Plane mehr "
            "Teilnehmende ein oder wähle eine stärkere Manipulation."
        )
    else:
        st.success(
            "Der Stichprobenplan ist ausreichend dimensioniert, um einen Effekt "
            "dieser Größe mit der konventionellen Power von 80 % zu entdecken."
        )

merkkasten(
    "Merke",
    "Ein SEM ist ein DAG mit Gleichungen: Es zerlegt <i>Wirkwege</i>, steht "
    "und fällt aber mit seinen Annahmen. Survey Experiments geben euch das "
    "Kostbarste der Kausalinferenz, nämlich <b>eigene Randomisierung</b>, "
    "für die Fragen, zu denen es keine Daten gibt. Die Power wird vor der "
    "Erhebung gerechnet, nicht danach.",
    typ="merke",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Wie misst man etwas, das man nicht direkt beobachten kann, also "
            "Vertrauen, Motivation oder politische Einstellung? Latente "
            "Konstrukte, Faktorenanalyse und Reliabilität sind das Handwerk "
            "dahinter."
        ),
        (
            "Wann ist ein SEM überhaupt schätzbar? Modelle mit zu vielen "
            "freien Parametern sind nicht identifiziert, und die Software sagt "
            "es einem nicht immer deutlich. Woran erkennt man es vorher?"
        ),
        (
            "Baut ein eigenes Survey-Experiment: Vignetten oder "
            "Conjoint-Design, Attention Checks, Randomisierung im Fragebogen. "
            "Legt vorher fest, was ihr testen wollt."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>semopy</code> (Python) oder <code>lavaan</code> "
        "(R), ein Fragebogen-Tool eurer Wahl, und die Power-Analyse aus "
        "der Vertiefung oben, um die nötige Teilnehmerzahl vorab zu "
        "bestimmen."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- R. B. Kline (2023), *Principles and Practice of Structural Equation Modeling*, 5. Aufl., Guilford Press
- D. C. Mutz (2011), *Population-Based Survey Experiments*, Princeton University Press
- J. Pearl (2009), *Causality: Models, Reasoning, and Inference*, 2. Aufl., Cambridge University Press, Kap. 5
"""
)

st.markdown("## Wie geht es weiter?")
weiter_dags, weiter_themen = st.columns(2)
with weiter_dags:
    st.page_link(
        "views/kausalitaet/dags_confounding.py",
        label="Grundlage: DAGs & Confounding",
        icon="🕸️",
    )
with weiter_themen:
    st.page_link(
        "views/projekte/themen.py",
        label="Zu den Themen der Gruppenarbeiten",
        icon="🧭",
    )
