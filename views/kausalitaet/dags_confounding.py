"""Kapitel Kausalität: DAGs & Confounding.

Kausale Graphen als Sprache für Annahmen, die drei Grundmuster
(Confounder, Mediator, Collider) interaktiv, Backdoor-Kriterium und ein
Vorgeschmack auf Causal Discovery.
"""

import numpy as np
import streamlit as st

from utils.theming import gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "🕸️",
    "DAGs & Confounding",
    "Eine Sprache für Ursachen und die Frage, wofür man adjustieren darf",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Im Kapitel „Korrelation ≠ Kausalität“ hat die Adjustierung für $Z$ die
Scheinkorrelation beseitigt. Offen blieb die Frage, welche Variablen in die
Regression gehören. Die Antwort erfordert ein Modell der Kausalstruktur, und
die kompakteste Sprache dafür sind **Directed Acyclic Graphs (DAGs)**:

- **Knoten** stehen für Variablen, **Pfeile** für direkte kausale Einflüsse.
- *Directed*: Pfeile haben eine Richtung, von der Ursache zur Wirkung.
- *Acyclic*: Es gibt keine Schleifen, nichts verursacht sich selbst über Umwege.

Ein DAG ist eine **Annahme über die Welt**, kein Ergebnis aus den Daten.
Genau darin liegt seine Stärke: Er zwingt dazu, die eigenen Annahmen
explizit aufzuzeichnen, und liefert dann rein mechanisch die Antwort,
wofür adjustiert werden muss.
"""
)

# --------------------------------- Demo 1: Die drei Grundmuster
st.markdown("## Demo: Confounder, Mediator, Collider")
st.markdown(
    r"""
Jeder noch so große kausale Graph setzt sich aus drei elementaren Mustern
zusammen:

| Muster | Struktur | Rolle von $W$ |
|---|---|---|
| Fork | $X \leftarrow W \rightarrow Y$ | Confounder |
| Chain | $X \rightarrow W \rightarrow Y$ | Mediator |
| Collider | $X \rightarrow W \leftarrow Y$ | Collider |

Wähle unten ein Muster und den wahren direkten Effekt $c$ von $X$ auf $Y$.
Verglichen werden zwei Auswertungen: eine einfache Regression von $Y$ auf $X$
(**ohne** Adjustierung) und eine multiple Regression von $Y$ auf $X$ und $W$
(**mit** Adjustierung). Das zentrale Ergebnis lautet: Adjustierung ist
**nicht in jeder Struktur korrekt**. Sie kann Verzerrung beseitigen, den
Estimand verändern oder Verzerrung überhaupt erst erzeugen.
"""
)

struktur = st.radio(
    "Struktur",
    ["Confounder (Fork)", "Mediator (Chain)", "Collider"],
    horizontal=True,
)
effekt_c = st.slider("c: direkter Effekt X → Y", 0.0, 1.0, 0.5, step=0.1)


@st.cache_data
def struktur_daten(struktur: str, c: float, n: int = 3000, seed: int = 11):
    rng = np.random.default_rng(seed)
    e = lambda: rng.normal(0, 1, n)  # noqa: E731
    if struktur.startswith("Confounder"):
        w = e()                    # W beeinflusst X und Y
        x = w + e()
        y = c * x + w + e()
    elif struktur.startswith("Mediator"):
        x = e()
        w = x + e()                # W liegt auf dem Pfad X → W → Y
        y = c * x + w + e()
    else:                          # Collider: X und Y beeinflussen W
        x = e()
        y = c * x + e()
        w = x + y + e()
    return x, w, y


x, w, y = struktur_daten(struktur, effekt_c)

ohne = np.polyfit(x, y, 1)[0]
design = np.column_stack([x, w, np.ones_like(x)])
mit = np.linalg.lstsq(design, y, rcond=None)[0][0]

DAGS = {
    "Confounder (Fork)": """
digraph { rankdir=LR; bgcolor="transparent";
  node [shape=circle, style=filled, fillcolor="#EEF3FA", color="#3E6DB5", fontname="sans-serif", fixedsize=true, width=0.7];
  edge [color="#5C6470"];
  W [fillcolor="#FCF1E9", color="#E8804C"];
  W -> X; W -> Y; X -> Y [color="#E8804C", penwidth=2];
}""",
    "Mediator (Chain)": """
digraph { rankdir=LR; bgcolor="transparent";
  node [shape=circle, style=filled, fillcolor="#EEF3FA", color="#3E6DB5", fontname="sans-serif", fixedsize=true, width=0.7];
  edge [color="#5C6470"];
  W [fillcolor="#FCF1E9", color="#E8804C"];
  X -> W; W -> Y; X -> Y [color="#E8804C", penwidth=2];
}""",
    "Collider": """
digraph { rankdir=LR; bgcolor="transparent";
  node [shape=circle, style=filled, fillcolor="#EEF3FA", color="#3E6DB5", fontname="sans-serif", fixedsize=true, width=0.7];
  edge [color="#5C6470"];
  W [fillcolor="#FCF1E9", color="#E8804C"];
  X -> W; Y -> W; X -> Y [color="#E8804C", penwidth=2];
}""",
}

spalte_dag, spalte_zahlen = st.columns([2, 3])
with spalte_dag:
    st.graphviz_chart(DAGS[struktur], use_container_width=True)
with spalte_zahlen:
    if struktur.startswith("Mediator"):
        wahrer_total = effekt_c + 1.0  # direkter Pfad + Pfad über W
        st.metric("Wahrer totaler Effekt (direkt + über W)", f"{wahrer_total:.2f}")
    else:
        st.metric("Wahrer Effekt X → Y", f"{effekt_c:.2f}")
    metrik_ohne, metrik_mit = st.columns(2)
    metrik_ohne.metric("Schätzung ohne Adjustierung", f"{ohne:.2f}")
    metrik_mit.metric("Schätzung mit Adjustierung für W", f"{mit:.2f}")

if struktur.startswith("Confounder"):
    st.success(
        "**Adjustierung hilft:** W ist ein Confounder, der Backdoor-Pfad "
        "X ← W → Y verzerrt die naive Schätzung. Mit Adjustierung trifft die "
        "Schätzung das wahre c."
    )
elif struktur.startswith("Mediator"):
    st.warning(
        "**Es kommt auf die Fragestellung an:** W liegt auf dem Wirkpfad. "
        "Ohne Adjustierung misst du den **totalen** Effekt (direkt plus über "
        "W), mit Adjustierung nur den **direkten** Pfad c. Wer den "
        "Gesamteffekt einer Maßnahme wissen will, darf den Mediator nicht "
        "kontrollieren."
    )
else:
    st.error(
        "**Adjustierung schadet:** Ohne Adjustierung stimmt die Schätzung. "
        "W ist ein Collider, X und Y verursachen W gemeinsam. Wer für W "
        "adjustiert, öffnet einen künstlichen Pfad zwischen X und Y und "
        "**erzeugt** Verzerrung, wo zuvor keine war (Collider Bias, auch "
        "Selection Bias)."
    )

st.markdown("## Das Backdoor-Kriterium")
st.markdown(
    r"""
Pfade, die mit einem Pfeil *in* die Treatment-Variable beginnen, etwa
$X \leftarrow W \rightarrow Y$, heißen **Backdoor Paths**. Sie transportieren
nicht-kausale Assoziation. Das **Backdoor Criterion** (Pearl) präzisiert,
wann eine Adjustierungsmenge den kausalen Effekt identifiziert:
"""
)

merkkasten(
    "Backdoor Criterion (Definition)",
    "Eine Variablenmenge S erfüllt das Backdoor Criterion relativ zu (X, Y), "
    "wenn (i) kein Element von S ein Nachfahre von X ist und (ii) S jeden "
    "Backdoor Path zwischen X und Y blockiert. Ein Pfad ist blockiert, wenn "
    "er eine Chain oder Fork enthält, deren Mittelknoten in S liegt, oder "
    "einen Collider, der samt seinen Nachfahren <i>nicht</i> in S liegt "
    "(d-Separation).",
    typ="definition",
)

st.markdown(
    r"""
Erfüllt $S$ das Backdoor Criterion, lässt sich der Interventionseffekt durch
beobachtbare Größen ausdrücken (**Adjustment Formula**):

$$
E\big[Y \mid \mathrm{do}(X = x)\big]
= \sum_{s} E\big[Y \mid X = x, S = s\big]\, P(S = s).
$$

Damit wird die Wahl der Kontrollvariablen zu einer mechanischen Prüfung am
Graphen. Bauchgefühl und die verbreitete Heuristik, mehr Kontrollvariablen
seien stets besser, werden überflüssig. Der Collider-Fall zeigt, warum diese
Heuristik gefährlich ist: Er tritt überall dort auf, wo **Selektion** im
Spiel ist. Untersucht man etwa nur zugelassene Studierende, wobei die
Zulassung ein Collider von Talent und Fleiß ist, entsteht innerhalb der
Auswahl eine negative Korrelation zwischen Talent und Fleiß, die in der
Gesamtbevölkerung nicht existiert (*Collider Bias*, *Sample Selection Bias*).
"""
)

# ------------------------------------- Demo 2: Causal Discovery Teaser
st.markdown("## Ausblick: Causal Discovery")
st.markdown(
    r"""
Bisher haben wir den DAG als gegeben **angenommen**. **Causal Discovery**
kehrt die Blickrichtung um: Algorithmen wie der **PC-Algorithmus** testen
systematisch bedingte Unabhängigkeiten, also Aussagen der Form
$X \perp Y \mid Z$, und schließen daraus, welche Graphstrukturen mit den
Daten vereinbar sind. Die folgenden Kennzahlen stammen aus einer Simulation
mit verborgener Struktur. Versuche, sie zu rekonstruieren:
"""
)


@st.cache_data
def discovery_daten(n: int = 4000, seed: int = 21):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)          # verborgene Wahrheit: Kette X → Z → Y
    z = x + rng.normal(0, 1, n)
    y = z + rng.normal(0, 1, n)
    return x, z, y


def partielle_korrelation(a, b, kontrolle):
    rest_a = a - np.polyval(np.polyfit(kontrolle, a, 1), kontrolle)
    rest_b = b - np.polyval(np.polyfit(kontrolle, b, 1), kontrolle)
    return np.corrcoef(rest_a, rest_b)[0, 1]


dx, dz, dy = discovery_daten()
korr_xy = np.corrcoef(dx, dy)[0, 1]
korr_xz = np.corrcoef(dx, dz)[0, 1]
korr_zy = np.corrcoef(dz, dy)[0, 1]
partiell_xy = partielle_korrelation(dx, dy, dz)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Korr(X, Y)", f"{korr_xy:.2f}")
k2.metric("Korr(X, Z)", f"{korr_xz:.2f}")
k3.metric("Korr(Z, Y)", f"{korr_zy:.2f}")
k4.metric("Korr(X, Y | Z)", f"{partiell_xy:.2f}", help="Korrelation von X und Y, nachdem Z herausgerechnet wurde")
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Dieses Kapitel setzt den DAG als gegeben voraus. <b>Causal "
            "Discovery</b> dreht die Frage um: Lässt sich die Struktur aus den "
            "Daten selbst lernen? Algorithmen wie PC, GES und NOTEARS "
            "versuchen genau das."
        ),
        (
            "Warum legen Daten allein den DAG nie vollständig fest: Mehrere "
            "Graphen erzeugen exakt dieselben bedingten Unabhängigkeiten "
            "(Markov-Äquivalenzklasse). Was bleibt dann übrig, und welche "
            "Zusatzannahmen lösen die Mehrdeutigkeit auf?"
        ),
        (
            "Was passiert, wenn ein wichtiger Confounder gar nicht erhoben "
            "wurde? Der FCI-Algorithmus ist dafür gebaut, versteckte Variablen "
            "zuzulassen. Wie viel Aussage bleibt dann noch übrig?"
        ),
    ],
    hinweis=(
        "Startpunkt: <code>causal-learn</code> (Implementierungen von PC, "
        "GES und FCI) und <code>dowhy</code>. Testet die Verfahren zuerst "
        "an simulierten Daten, deren wahren DAG ihr kennt, sonst könnt ihr "
        "die Ergebnisse nicht bewerten."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- J. Pearl, M. Glymour & N. P. Jewell (2016), *Causal Inference in Statistics: A Primer*, Wiley, Kap. 2 und 3
- S. Cunningham (2021), *Causal Inference: The Mixtape*, Yale University Press, Kap. 3 (frei online)
- J. Peters, D. Janzing & B. Schölkopf (2017), *Elements of Causal Inference*, MIT Press (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_po, weiter_sem = st.columns(2)
with weiter_po:
    st.page_link(
        "views/kausalitaet/potential_outcomes.py",
        label="Weiter: Potential Outcomes & RCTs",
        icon="⚖️",
    )
with weiter_sem:
    st.page_link(
        "views/kausalitaet/sem_surveys.py",
        label="Verwandt: SEMs & Survey Experiments",
        icon="📋",
    )
