"""Musterkapitel Kausalität: Korrelation ≠ Kausalität.

Interaktive Einführung: Scheinkorrelation durch Confounder und das
Simpson-Paradox.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "🔀",
    "Korrelation ≠ Kausalität",
    "Warum Muster in Daten noch keine Ursachen sind",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Ein paar Zusammenhänge, die sich in echten Daten finden lassen:

- In Regionen mit mehr **Störchen** werden mehr **Kinder** geboren.
- An Tagen mit hohem **Eisverkauf** gibt es mehr **Sonnenbrände**.
- Menschen, die **Rotwein** trinken, sind im Schnitt gesünder.

Alle drei Zusammenhänge sind statistisch real, und doch ist keiner von ihnen
ein kausaler Effekt. Hinter jedem steht eine **dritte Variable**, die beide
Größen gemeinsam beeinflusst: Ländlichkeit, Sonnenstunden, Einkommen. Solche
Variablen heißen **Confounder**.
"""
)

st.markdown(
    r"""
Der Unterschied lässt sich präzise fassen. Korrelation ist eine
**symmetrische** Eigenschaft der gemeinsamen Verteilung zweier
Zufallsvariablen. Ein kausaler Effekt ist dagegen eine Aussage über
**Interventionen**: Wie ändert sich $Y$, wenn wir $X$ aktiv setzen? In der
Notation von Pearl unterscheidet man entsprechend die *Beobachtung*
$E[Y \mid X = x]$ von der *Intervention* $E[Y \mid \mathrm{do}(X = x)]$.
Confounding ist genau die Situation, in der beide Größen auseinanderfallen.
"""
)

merkkasten(
    "Definition",
    "<b>Korrelation:</b> Zwei Größen bewegen sich systematisch gemeinsam. "
    "<b>Kausalität:</b> Ein Eingriff in die eine Größe <i>verändert</i> die "
    "andere. Ein Confounder Z, der beide beeinflusst, erzeugt Korrelation "
    "ganz ohne Kausalität.",
    typ="definition",
)

# ------------------------------------------ Demo 1: Confounder-Simulator
st.markdown("## Demo: Der Confounder-Simulator")
st.markdown(
    r"""
Wir simulieren die Eis-Sonnenbrand-Welt über ein kleines **Strukturmodell**:
Die Sonnenstunden $Z$ beeinflussen den Eisverkauf $X$ und die Sonnenbrände
$Y$, zusätzlich lässt das Modell einen direkten Effekt von $X$ auf $Y$ zu:

$$
X = a\,Z + \varepsilon_X, \qquad
Y = c\,X + b\,Z + \varepsilon_Y, \qquad
\varepsilon_X, \varepsilon_Y \sim \mathcal{N}(0, 1)
$$

Der Parameter $c$ ist der wahre kausale Effekt. Setze $c = 0$ und beobachte,
welchen Zusammenhang eine naive Auswertung dennoch ausweist.
"""
)

regler_a, regler_b, regler_c = st.columns(3)
effekt_zx = regler_a.slider("a: Sonne → Eisverkauf", 0.0, 2.0, 1.2, step=0.1)
effekt_zy = regler_b.slider("b: Sonne → Sonnenbrände", 0.0, 2.0, 1.2, step=0.1)
effekt_xy = regler_c.slider(
    "c: Eisverkauf → Sonnenbrände (wahrer kausaler Effekt)",
    -1.0, 1.0, 0.0, step=0.1,
)


@st.cache_data
def confounder_daten(a: float, b: float, c: float, n: int = 400, seed: int = 2026):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    x = a * z + rng.normal(0, 1, n)
    y = c * x + b * z + rng.normal(0, 1, n)
    return z, x, y


z, x, y = confounder_daten(effekt_zx, effekt_zy, effekt_xy)

# Naive Schätzung: einfache Regression y ~ x (ignoriert Z)
naiv = np.polyfit(x, y, 1)[0]
# Adjustierte Schätzung: Regression y ~ x + z, Koeffizient auf x
design = np.column_stack([x, z, np.ones_like(x)])
adjustiert = np.linalg.lstsq(design, y, rcond=None)[0][0]

spalte_dag, spalte_streu = st.columns([2, 3])

with spalte_dag:
    st.graphviz_chart(
        f"""
digraph {{
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#EEF3FA",
          color="#3E6DB5", fontname="sans-serif"];
    edge [fontname="sans-serif", color="#5C6470"];
    Z [label="☀️ Sonnenstunden (Z)"];
    X [label="🍦 Eisverkauf (X)"];
    Y [label="🔥 Sonnenbrände (Y)"];
    Z -> X [label="a = {effekt_zx:.1f}"];
    Z -> Y [label="b = {effekt_zy:.1f}"];
    X -> Y [label="c = {effekt_xy:.1f}", color="#E8804C", fontcolor="#B05A2B",
            penwidth=2];
}}
""",
        use_container_width=True,
    )

with spalte_streu:
    raster = np.linspace(x.min(), x.max(), 50)
    fig = go.Figure()
    fig.add_scatter(
        x=x, y=y, mode="markers", name="Tage",
        marker=dict(
            color=z, colorscale=[[0, "#D9E4F2"], [1, FARBEN["nacht"]]],
            size=7, opacity=0.75,
            colorbar=dict(title="Sonnenstunden Z", thickness=12),
        ),
    )
    fig.add_scatter(
        x=raster, y=np.polyval(np.polyfit(x, y, 1), raster), mode="lines",
        name=f"Naive Regressionsgerade (Steigung {naiv:.2f})",
        line=dict(color=FARBEN["sonne"], width=3),
    )
    fig.update_layout(
        xaxis_title="Eisverkauf X", yaxis_title="Sonnenbrände Y", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

metrik_wahr, metrik_naiv, metrik_adj = st.columns(3)
metrik_wahr.metric("Wahrer Effekt c", f"{effekt_xy:.2f}")
metrik_naiv.metric(
    "Naive Schätzung (y ~ x)", f"{naiv:.2f}", delta=f"{naiv - effekt_xy:+.2f} verzerrt",
    delta_color="inverse",
)
metrik_adj.metric(
    "Adjustiert für Z (y ~ x + z)", f"{adjustiert:.2f}",
    delta=f"{adjustiert - effekt_xy:+.2f}", delta_color="inverse",
)

if abs(effekt_xy) < 0.05 and abs(naiv) > 0.3:
    merkkasten(
        "Spurious Correlation",
        f"Der wahre Effekt beträgt <b>{effekt_xy:.1f}</b>, Eis verursacht "
        f"hier also keinen Sonnenbrand. Dennoch weist die naive Auswertung "
        f"eine Steigung von <b>{naiv:.2f}</b> aus. Der gemeinsame Treiber Z "
        "erzeugt den scheinbaren Zusammenhang. Die Regression, die Z "
        "berücksichtigt, findet dagegen einen Wert nahe null.",
        typ="achtung",
    )

st.markdown(
    r"""
Die Färbung der Punkte zeigt den Mechanismus: Sonnige Tage (dunkel) liegen
rechts oben, trübe Tage (hell) links unten. Die naive Regressionsgerade
vergleicht in Wahrheit Wettertypen. Die Verzerrung lässt sich exakt angeben
und heißt **Omitted Variable Bias**: Die einfache Regression von $Y$ auf $X$
schätzt nicht $c$, sondern

$$
\beta_{\text{naiv}}
= c + b\,\frac{\mathrm{Cov}(X, Z)}{\mathrm{Var}(X)}
= c + \underbrace{\frac{a\,b}{a^2 + 1}}_{\text{Bias-Term}} .
$$

Der Bias-Term verschwindet genau dann, wenn $a = 0$ oder $b = 0$ gilt, wenn
also $Z$ nicht beide Variablen beeinflusst. Die multiple Regression von $Y$
auf $X$ und $Z$, die Adjustierung für $Z$, blockiert diesen Pfad und trifft
$c$ bis auf den Schätzfehler.
"""
)

# --------------------------------------------- Demo 2: Simpson-Paradox
st.markdown("## Demo: Das Simpson-Paradox")
st.markdown(
    """
Confounder können Zusammenhänge nicht nur vortäuschen, sondern im Vorzeichen
**umkehren**. Ein klassisches Beispiel: In den aggregierten Daten einer
Beobachtungsstudie geht mehr **Sport** mit *höherem* **Cholesterin** einher.
Schlüssle die Daten nach **Altersgruppen** auf und vergleiche die Steigungen
innerhalb der Gruppen mit dem Gesamttrend.
"""
)


@st.cache_data
def simpson_daten(seed: int = 7):
    rng = np.random.default_rng(seed)
    gruppen = {
        "20–35 Jahre": (2.0, 195.0),
        "40–55 Jahre": (5.0, 215.0),
        "60–75 Jahre": (8.0, 235.0),
    }
    frames = []
    for name, (sport_mitte, chol_mitte) in gruppen.items():
        n = 60
        sport = np.clip(rng.normal(sport_mitte, 1.2, n), 0, None)
        # Innerhalb jeder Altersgruppe senkt Sport das Cholesterin.
        chol = chol_mitte - 4.0 * (sport - sport_mitte) + rng.normal(0, 8, n)
        frames.append((name, sport, chol))
    return frames


frames = simpson_daten()
alle_sport = np.concatenate([f[1] for f in frames])
alle_chol = np.concatenate([f[2] for f in frames])
gesamt_steigung = np.polyfit(alle_sport, alle_chol, 1)[0]

aufschluesseln = st.toggle("Nach Altersgruppen aufschlüsseln")

fig_simpson = go.Figure()
gruppen_farben = [FARBEN["gletscher"], FARBEN["sonne"], FARBEN["wiese"]]

if not aufschluesseln:
    fig_simpson.add_scatter(
        x=alle_sport, y=alle_chol, mode="markers", name="Alle Personen",
        marker=dict(color=FARBEN["schiefer"], size=7, opacity=0.6),
    )
    raster = np.linspace(alle_sport.min(), alle_sport.max(), 50)
    fig_simpson.add_scatter(
        x=raster, y=np.polyval(np.polyfit(alle_sport, alle_chol, 1), raster),
        mode="lines", name=f"Gesamttrend (Steigung {gesamt_steigung:+.1f})",
        line=dict(color=FARBEN["beere"], width=4),
    )
else:
    for (name, sport, chol), farbe in zip(frames, gruppen_farben):
        steigung = np.polyfit(sport, chol, 1)[0]
        fig_simpson.add_scatter(
            x=sport, y=chol, mode="markers", name=f"{name}",
            marker=dict(color=farbe, size=7, opacity=0.7),
        )
        raster = np.linspace(sport.min(), sport.max(), 20)
        fig_simpson.add_scatter(
            x=raster, y=np.polyval(np.polyfit(sport, chol, 1), raster),
            mode="lines", name=f"{name}: Steigung {steigung:+.1f}",
            line=dict(color=farbe, width=3),
        )

fig_simpson.update_layout(
    xaxis_title="Sport (Stunden pro Woche)",
    yaxis_title="Cholesterin (mg/dl)",
    height=440,
)
st.plotly_chart(fig_simpson, use_container_width=True)

if aufschluesseln:
    st.success(
        "**Auflösung:** Innerhalb *jeder* Altersgruppe senkt Sport das "
        "Cholesterin (negative Steigungen). Der positive Gesamttrend entsteht "
        "nur, weil Ältere sowohl mehr Sport treiben als auch höheres "
        "Cholesterin haben. Das **Alter** ist der Confounder."
    )
else:
    st.info(
        f"Der Gesamttrend hat die Steigung **{gesamt_steigung:+.1f}**, mehr "
        "Sport geht also scheinbar mit höherem Cholesterin einher. Aktiviere "
        "die Aufschlüsselung, bevor du dieses Ergebnis interpretierst."
    )

merkkasten(
    "Merke",
    "Ob ein Zusammenhang erscheint, verschwindet oder sich umkehrt, kann "
    "davon abhängen, welche Gruppen man betrachtet (<b>Simpson's Paradox</b>). "
    "Welche Aufteilung die richtige ist, verraten die Daten allein nicht. "
    "Dafür braucht es Annahmen über die Kausalstruktur.",
    typ="merke",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Das Thema <b>Examples und Case Studies</b> beginnt hier: Sucht "
            "euch ein bekanntes Paper mit offenem Replikationspaket und "
            "rechnet es nach. Kommen dieselben Zahlen heraus? Falls nicht, "
            "liegt es an den Daten, am Code oder an der Beschreibung im Text?"
        ),
        (
            "Welche einzelne Annahme trägt das Ergebnis? Variiert sie und "
            "schaut, ab wann der Befund kippt. Ein Ergebnis, das erst bei "
            "grober Verletzung verschwindet, ist etwas ganz anderes als eines, "
            "das schon bei kleinen Änderungen umfällt."
        ),
        (
            "Confounder und Simpson-Paradox sind hier simuliert. Findet ein "
            "reales Beispiel, in dem aggregierter und gruppenweiser "
            "Zusammenhang gegenläufig sind, und begründet, welche der beiden "
            "Ebenen die Frage beantwortet."
        ),
    ],
    hinweis=(
        "Startpunkt: Harvard Dataverse und das AEA Data and Code "
        "Repository für Replikationspakete. Das Beispielprojekt unter "
        "Gruppenprojekte zeigt, wie die fertige Arbeit aufgebaut sein kann."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- S. Cunningham (2021), *Causal Inference: The Mixtape*, Yale University Press, Kap. 1 und 4 (frei online)
- M. Facure, *Causal Inference for the Brave and True*, Kap. 1 und 2 (frei online)
- J. Pearl & D. Mackenzie (2018), *The Book of Why*, Basic Books
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    r"""
Die Adjustierung für $Z$ hat im Simulator funktioniert. Offen bleibt die
zentrale Frage, welche Variablen in die Adjustierung gehören und welche man
bewusst weglassen muss. Die Antwort liefert die Sprache der
**Directed Acyclic Graphs (DAGs)**.
"""
)
st.page_link(
    "views/kausalitaet/dags_confounding.py",
    label="Weiter: DAGs & Confounding",
    icon="🕸️",
)
