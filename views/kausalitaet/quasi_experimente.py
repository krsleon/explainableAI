"""Kapitel Kausalität: Quasi-Experimente (Difference-in-Differences & RDD).

Kausale Effekte ohne Experiment: DiD mit Parallel-Trends-Simulation und
Regression Discontinuity Design mit Bandwidth-Wahl.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "📐",
    "Quasi-Experimente: DiD & RDD",
    "Wenn niemand würfeln darf: kausale Effekte aus natürlichen Experimenten",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Mindestlohn, Rauchverbote, Schulreformen: Für die großen Politikfragen kann
niemand einen RCT durchführen, denn man kann Regionen nicht per Münzwurf
einen Mindestlohn zuweisen. Die Ökonometrie behilft sich mit
**Quasi-Experimenten**: Situationen, in denen die Welt einem Experiment sehr
nahe kommt, sofern man sie geschickt auswertet. Die zwei bekanntesten
Designs lernst du in diesem Kapitel kennen.
"""
)

# ---------------------------------------------- Demo 1: DiD
st.markdown("## Demo: Difference-in-Differences (DiD)")
st.markdown(
    r"""
Die klassische Anwendung stammt von Card & Krueger (1994): New Jersey erhöht
den **Mindestlohn**, das benachbarte Pennsylvania nicht. Ein reiner
Vorher-nachher-Vergleich enthielte den allgemeinen Zeittrend, ein reiner
Gruppenvergleich die strukturellen Unterschiede der Bundesstaaten. DiD
kombiniert deshalb beide Differenzen. Mit Behandlungsgruppe $k$ und Kontrollgruppe
$u$ lautet der Schätzer im $2 \times 2$-Design:

$$
\widehat{\delta}^{\,2\times 2}
= \big(\bar{y}_k^{\,\text{post}} - \bar{y}_k^{\,\text{pre}}\big)
- \big(\bar{y}_u^{\,\text{post}} - \bar{y}_u^{\,\text{pre}}\big).
$$

In Potential-Outcomes-Notation zerlegt er sich in

$$
\widehat{\delta}^{\,2\times 2}
= \mathrm{ATT}
+ \underbrace{\big[E[Y_k^0 \mid \text{Post}] - E[Y_k^0 \mid \text{Pre}]\big]
- \big[E[Y_u^0 \mid \text{Post}] - E[Y_u^0 \mid \text{Pre}]\big]}_{\text{Bias durch nicht-parallele Trends}}.
$$

Damit ist die Identifikationsannahme sichtbar: **Parallel Trends**. Ohne
Behandlung hätten sich beide Gruppen im Mittel gleich entwickelt, und der
Bias-Term wäre null. Die Annahme betrifft ein Kontrafaktual und ist deshalb
nicht testbar, üblich ist die Prüfung der *Pre-Trends* als
Plausibilitätsindiz. Untersuche in der Simulation, was bei einer Verletzung
geschieht.
"""
)

regler_tau, regler_trend = st.columns(2)
did_effekt = regler_tau.slider("Wahrer Reformeffekt", -10.0, 10.0, 5.0, step=0.5)
trend_bruch = regler_trend.slider(
    "Verletzung der Parallel Trends (Extra-Trend der Reformregion pro Jahr)",
    -2.0, 2.0, 0.0, step=0.25,
)

JAHRE = np.arange(2015, 2025)
REFORM_JAHR = 2020


@st.cache_data
def did_daten(tau: float, extra_trend: float, seed: int = 6):
    rng = np.random.default_rng(seed)
    t = JAHRE - JAHRE[0]
    kontrolle = 60 + 1.5 * t + rng.normal(0, 0.8, len(t))
    nach_reform = (JAHRE >= REFORM_JAHR).astype(float)
    behandelt = (
        52 + 1.5 * t + extra_trend * t + tau * nach_reform + rng.normal(0, 0.8, len(t))
    )
    return kontrolle, behandelt


kontrolle, behandelt = did_daten(did_effekt, trend_bruch)
vor = JAHRE < REFORM_JAHR

# DiD-Schätzer: (Δ Behandlungsgruppe) − (Δ Kontrollgruppe)
did_schaetzer = (behandelt[~vor].mean() - behandelt[vor].mean()) - (
    kontrolle[~vor].mean() - kontrolle[vor].mean()
)

# Kontrafaktual: Behandlungsgruppe wäre wie die Kontrolle weitergelaufen
kontrafaktual = behandelt[vor].mean() + (kontrolle - kontrolle[vor].mean())

fig_did = go.Figure()
fig_did.add_scatter(
    x=JAHRE, y=kontrolle, mode="lines+markers", name="Kontrollregion",
    line=dict(color=FARBEN["gletscher"], width=3),
)
fig_did.add_scatter(
    x=JAHRE, y=behandelt, mode="lines+markers", name="Reformregion",
    line=dict(color=FARBEN["sonne"], width=3),
)
fig_did.add_scatter(
    x=JAHRE[~vor], y=kontrafaktual[~vor], mode="lines",
    name="Kontrafaktual (ohne Reform, laut Parallel Trends)",
    line=dict(color=FARBEN["schiefer"], width=2, dash="dash"),
)
fig_did.add_vline(
    x=REFORM_JAHR - 0.5, line_dash="dot", line_color=FARBEN["beere"],
    annotation_text="Reform", annotation_position="top",
)
fig_did.update_layout(
    xaxis_title="Jahr", yaxis_title="Beschäftigung (Index)", height=430,
)
st.plotly_chart(fig_did, use_container_width=True)

metrik_wahr, metrik_did = st.columns(2)
metrik_wahr.metric("Wahrer Effekt", f"{did_effekt:+.1f}")
metrik_did.metric(
    "DiD-Schätzung", f"{did_schaetzer:+.1f}",
    delta=f"{did_schaetzer - did_effekt:+.1f}", delta_color="inverse",
)

if abs(trend_bruch) > 0.01:
    st.error(
        "**Parallel Trends verletzt:** Die Reformregion folgt bereits vor der "
        "Reform einem eigenen Trend, und der DiD-Estimator rechnet diesen Trend "
        "fälschlich dem Reformeffekt zu. In den **Vor-Reform-Jahren** ist die "
        "Abweichung sichtbar: Nicht-parallele Pre-Trends sind ein deutliches "
        "Warnsignal für die Validität des Designs."
    )
else:
    st.success(
        "**Parallel Trends erfüllt:** Beide Regionen entwickeln sich vor der "
        "Reform parallel, und der Abstand zwischen Reformregion und "
        "gestricheltem Kontrafaktual entspricht dem kausalen Effekt."
    )

st.markdown(
    r"""
In der Praxis wird DiD als Regression geschätzt:

$$
y_{it} = \alpha + \gamma\, G_i + \lambda\, \mathrm{Post}_t
+ \delta\, (G_i \times \mathrm{Post}_t) + \varepsilon_{it},
$$

wobei $G_i$ die Gruppenzugehörigkeit und $\mathrm{Post}_t$ die Nachperiode
markiert. Der Koeffizient $\delta$ des Interaktionsterms ist der
DiD-Schätzer. Die Regressionsform erlaubt Kontrollvariablen, mehrere
Perioden und Gruppen (Zweiwege-Fixed-Effects) sowie korrekte Standardfehler.
"""
)

# ---------------------------------------------- Demo 2: RDD
with vertiefung("Regression Discontinuity Design (RDD)"):
    st.markdown(
        r"""
    Das zweite Design nutzt Zuteilungsregeln mit **hartem Schwellenwert**.
    Beispiel: Ein Stipendium wird ab 60 Punkten im Auswahltest vergeben. Die
    Punktzahl ist die **Running Variable** $X_i$, der Schwellenwert der
    **Cutoff** $c_0$. Im *sharp design* ist das Treatment deterministisch:

    $$
    D_i = \mathbb{1}\{X_i \geq c_0\}.
    $$

    Personen knapp unter und knapp über $c_0$ sind in allen relevanten Merkmalen
    praktisch identisch, lokal wirkt die Zuteilung also wie randomisiert. Der
    RDD-Estimand ist der Effekt am Cutoff:

    $$
    \tau_{c_0} = E\big[Y^1 - Y^0 \mid X = c_0\big]
    = \lim_{x \,\downarrow\, c_0} E[Y \mid X = x]
    - \lim_{x \,\uparrow\, c_0} E[Y \mid X = x].
    $$

    Geschätzt wird der Sprung zweier lokaler Regressionen. Die zentrale
    Identifikationsannahme ist die **Stetigkeit** der erwarteten potenziellen
    Outcomes am Cutoff: Ohne Treatment gäbe es dort keinen Sprung. Insbesondere
    darf niemand seine Position relativ zu $c_0$ präzise manipulieren können.
    """
    )

    regler_rdd_tau, regler_h, regler_kruemmung = st.columns(3)
    rdd_effekt = regler_rdd_tau.slider("Wahrer Stipendieneffekt (Tsd. € Einkommen)", 0.0, 15.0, 6.0, step=0.5)
    bandbreite = regler_h.slider("Bandbreite um den Cutoff (± Punkte)", 5, 40, 15)
    kruemmung = regler_kruemmung.slider("Nichtlinearität des Hintergrundtrends", 0.0, 1.0, 0.5, step=0.1)

    CUTOFF = 60


    @st.cache_data
    def rdd_daten(tau: float, kruemmung: float, n: int = 700, seed: int = 12):
        rng = np.random.default_rng(seed)
        punkte = rng.uniform(20, 100, n)
        stipendium = (punkte >= CUTOFF).astype(float)
        einkommen = (
            35
            + 0.25 * (punkte - CUTOFF)
            + kruemmung * 12 * ((punkte - CUTOFF) / 40) ** 2
            + tau * stipendium
            + rng.normal(0, 3, n)
        )
        return punkte, einkommen


    punkte, einkommen = rdd_daten(rdd_effekt, kruemmung)

    links = (punkte >= CUTOFF - bandbreite) & (punkte < CUTOFF)
    rechts = (punkte >= CUTOFF) & (punkte <= CUTOFF + bandbreite)

    fit_links = np.polyfit(punkte[links], einkommen[links], 1)
    fit_rechts = np.polyfit(punkte[rechts], einkommen[rechts], 1)
    rdd_schaetzer = np.polyval(fit_rechts, CUTOFF) - np.polyval(fit_links, CUTOFF)

    fig_rdd = go.Figure()
    fig_rdd.add_vrect(
        x0=CUTOFF - bandbreite, x1=CUTOFF + bandbreite,
        fillcolor=FARBEN["nebel"], opacity=0.6, line_width=0,
    )
    fig_rdd.add_scatter(
        x=punkte[punkte < CUTOFF], y=einkommen[punkte < CUTOFF], mode="markers",
        name="kein Stipendium", marker=dict(color=FARBEN["gletscher"], size=5, opacity=0.45),
    )
    fig_rdd.add_scatter(
        x=punkte[punkte >= CUTOFF], y=einkommen[punkte >= CUTOFF], mode="markers",
        name="Stipendium", marker=dict(color=FARBEN["sonne"], size=5, opacity=0.45),
    )
    raster_links = np.linspace(CUTOFF - bandbreite, CUTOFF, 20)
    raster_rechts = np.linspace(CUTOFF, CUTOFF + bandbreite, 20)
    fig_rdd.add_scatter(
        x=raster_links, y=np.polyval(fit_links, raster_links), mode="lines",
        name="lokaler Fit links", line=dict(color=FARBEN["nacht"], width=4),
    )
    fig_rdd.add_scatter(
        x=raster_rechts, y=np.polyval(fit_rechts, raster_rechts), mode="lines",
        name="lokaler Fit rechts", line=dict(color=FARBEN["beere"], width=4),
    )
    fig_rdd.add_vline(x=CUTOFF, line_dash="dot", line_color=FARBEN["schiefer"])
    fig_rdd.update_layout(
        xaxis_title="Punkte im Auswahltest (Cutoff = 60)",
        yaxis_title="Späteres Einkommen (Tsd. €)", height=440,
    )
    st.plotly_chart(fig_rdd, use_container_width=True)

    metrik_wahr_rdd, metrik_rdd = st.columns(2)
    metrik_wahr_rdd.metric("Wahrer Effekt", f"{rdd_effekt:+.1f}")
    metrik_rdd.metric(
        "RDD-Schätzung (Sprung am Cutoff)", f"{rdd_schaetzer:+.1f}",
        delta=f"{rdd_schaetzer - rdd_effekt:+.1f}", delta_color="inverse",
    )

    st.markdown(
        r"""
    Variiere die **Bandwidth** $h$: Eine kleine Bandwidth nutzt wenige
    Beobachtungen (hohe Varianz), eine große verletzt die lokale lineare
    Approximation des gekrümmten Hintergrundtrends (Bias). Dieser
    Bias-Variance-Tradeoff, strukturell derselbe wie bei der Wahl der
    Modellkomplexität im ML-Kapitel, ist die zentrale praktische Entscheidung
    jeder RDD-Analyse, moderne Verfahren wählen $h$ datengetrieben.

    Ergänzend zum *Sharp Design*: Springt am Cutoff lediglich die
    Treatment-**Wahrscheinlichkeit** (*Fuzzy Design*), wird der Sprung als
    Instrument verwendet. Identifiziert wird dann, analog zum
    Non-Compliance-Fall im RCT-Kapitel, ein LATE für die Complier am Cutoff.
    """
    )

merkkasten(
    "Die Identifikationsannahmen",
    "<b>DiD:</b> Ohne Behandlung wären beide Gruppen parallel verlaufen. "
    "Diese Annahme ist nicht testbar, die Pre-Trends sollten aber stets "
    "geprüft werden. <b>RDD:</b> Niemand kann seine Position am Cutoff "
    "präzise manipulieren, und alles andere ändert sich am Cutoff stetig. "
    "Quasi-Experimente ersetzen den Münzwurf durch <i>Annahmen</i>. Die "
    "Kunst besteht darin, Designs zu finden, in denen diese Annahmen "
    "glaubwürdig sind.",
    typ="definition",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Was passiert, wenn Regionen zu <b>unterschiedlichen "
            "Zeitpunkten</b> behandelt werden? Der klassische "
            "Zweiwege-Fixed-Effects-Schätzer wird dann verzerrt, weil er "
            "bereits behandelte Einheiten als Kontrollgruppe missbraucht. Was "
            "schlagen Goodman-Bacon (2021) und Callaway &amp; Sant’Anna (2021) "
            "stattdessen vor?"
        ),
        (
            "Wie prüft man Pre-Trends, ohne sich selbst zu betrügen? Ein "
            "Event-Study-Plot mit weiten Konfidenzintervallen zeigt keine "
            "Verletzung, aber er belegt auch keine Parallelität. Wie "
            "unterscheidet man beides?"
        ),
        (
            "Wie wählt man die RDD-Bandbreite datengetrieben, statt sie wie in "
            "der Vertiefung per Schieberegler zu raten? Und wie testet man, ob "
            "Menschen ihre Position am Cutoff manipuliert haben?"
        ),
    ],
    hinweis=(
        "Startpunkt: <code>differences</code> und "
        "<code>linearmodels</code> (Python) für staggered DiD, "
        "<code>rdrobust</code> für Bandbreitenwahl und McCrary-Dichtetest. "
        "Ein echter Politikwechsel mit Paneldaten ist lehrreicher als jede "
        "Simulation."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- S. Cunningham (2021), *Causal Inference: The Mixtape*, Yale University Press, Kap. 6 und 9 (frei online)
- J. D. Angrist & J.-S. Pischke (2009), *Mostly Harmless Econometrics*, Princeton University Press, Kap. 5 und 6
- D. Card & A. B. Krueger (1994), *Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania*, American Economic Review 84(4), 772–793
"""
)

st.markdown("## Wie geht es weiter?")
weiter_dml, weiter_po = st.columns(2)
with weiter_dml:
    st.page_link(
        "views/kausalitaet/kausales_ml.py",
        label="Weiter: Kausales Machine Learning",
        icon="🎯",
    )
with weiter_po:
    st.page_link(
        "views/kausalitaet/potential_outcomes.py",
        label="Grundlage: Potential Outcomes & RCTs",
        icon="⚖️",
    )
