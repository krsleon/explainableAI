"""Kapitel ML: LLMs & kausales Denken.

Next-Token-Prinzip mit einem Mini-Sprachmodell zum Ausprobieren, dann die
Frage, ob Sprachmodelle kausal schließen können oder nur davon erzählen.
"""

from collections import Counter, defaultdict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "💬",
    "LLMs & kausales Denken",
    "Vom Autocomplete zur Weltbeschreibung, und wo die Grenze verläuft",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Ein **Large Language Model (LLM)** tut im Kern eine einzige Sache: Es sagt
das **nächste Token** vorher, gegeben den bisherigen Text. Das klingt
zunächst banal. Trainiert auf Billionen von Wörtern mit Milliarden von
Parametern entsteht daraus jedoch ein System, das Essays schreibt, Code
erklärt und Prüfungen besteht.

Um zu verstehen, was das bedeutet (und was nicht), bauen wir das Prinzip in
seiner einfachsten Form nach: ein **Bigram-Modell**, das ausschließlich die
Frage beantwortet, welches Wort statistisch auf das aktuelle folgt, die
einfachste Urform heutiger Sprachmodelle.
"""
)

# ------------------------------------------ Demo: Mini-Sprachmodell
st.markdown("## Demo: Ein Sprachmodell aus 20 Sätzen")

KORPUS = [
    "die sonne scheint über den bergen",
    "die sonne scheint auf den see",
    "die sonne geht hinter den bergen unter",
    "der hahn kräht am frühen morgen",
    "der hahn kräht und die sonne geht auf",
    "am morgen geht die sonne auf",
    "wir wandern über den grat zum gipfel",
    "wir wandern am see entlang zum dorf",
    "der weg zum gipfel ist steil und schön",
    "der weg zum dorf führt über die wiese",
    "auf der wiese blühen viele blumen",
    "viele blumen blühen am wegesrand",
    "im dorf gibt es eis und kuchen",
    "es gibt eis am see und im dorf",
    "das eis schmeckt am besten nach der wanderung",
    "nach der wanderung schwimmen wir im see",
    "der see ist kalt und klar",
    "die berge sind hoch und der see ist tief",
    "am abend sehen wir die sterne über den bergen",
    "die sterne leuchten hell über dem dorf",
]

with st.expander("Der komplette „Trainingsdatensatz“ (20 Sätze)"):
    st.markdown("\n".join(f"- {satz}" for satz in KORPUS))

st.markdown(
    r"""
Formal modelliert ein Sprachmodell die bedingte Verteilung des nächsten
Tokens gegeben den bisherigen Kontext, $p(w_t \mid w_1, \dots, w_{t-1})$.
Unser Bigram-Modell reduziert den Kontext auf das letzte Wort und schätzt
die Übergangswahrscheinlichkeiten aus Häufigkeiten. Beim Generieren wird
mit **Temperature** $T$ gezogen:

$$
p_T(w_{t+1} = w \mid w_t) \;\propto\; n(w_t, w)^{1/T},
$$

wobei $n(w_t, w)$ die Häufigkeit des Wortpaars im Korpus bezeichnet. Für
$T \to 0$ wird stets das häufigste Folgewort gewählt, während große $T$ die
Wahrscheinlichkeit gleichmäßiger verteilen. Dasselbe Prinzip steuert den
Sampling-Parameter Temperature realer LLMs, dort ersetzt ein tiefes
Transformer-Netz mit langem Kontext die Zähltabelle.
"""
)


@st.cache_data
def bigram_modell():
    folgen: dict[str, Counter] = defaultdict(Counter)
    for satz in KORPUS:
        woerter = satz.split() + ["■"]  # ■ = Satzende
        for aktuell, naechstes in zip(woerter, woerter[1:]):
            folgen[aktuell][naechstes] += 1
    return dict(folgen)


FOLGEN = bigram_modell()


def naechstes_wort_verteilung(wort: str, temperatur: float):
    kandidaten = FOLGEN.get(wort)
    if not kandidaten:
        return [], np.array([])
    woerter = list(kandidaten.keys())
    zaehler = np.array([kandidaten[w] for w in woerter], dtype=float)
    gewichte = zaehler ** (1.0 / max(temperatur, 1e-6))
    return woerter, gewichte / gewichte.sum()


regler_start, regler_temp = st.columns(2)
startwort = regler_start.selectbox(
    "Startwort", sorted(w for w in FOLGEN if w != "■"), index=None,
    placeholder="Wort wählen …",
)
temperatur = regler_temp.slider(
    "Temperature (Zufälligkeit)", 0.1, 2.0, 1.0, step=0.1,
    help="Niedrig: fast immer das häufigste Wort. Hoch: auch seltene Fortsetzungen bekommen eine Chance.",
)

if "llm_seed" not in st.session_state:
    st.session_state.llm_seed = 0

if startwort:
    woerter, probs = naechstes_wort_verteilung(startwort, temperatur)
    reihenfolge = np.argsort(probs)
    fig_probs = go.Figure(
        go.Bar(
            x=probs[reihenfolge],
            y=[woerter[i].replace("■", "⟨Satzende⟩") for i in reihenfolge],
            orientation="h",
            marker_color=FARBEN["gletscher"],
        )
    )
    fig_probs.update_layout(
        title=f"Wahrscheinlichkeit für das Wort nach „{startwort}“",
        xaxis_title="Wahrscheinlichkeit",
        height=max(220, 40 * len(woerter) + 120),
        margin=dict(t=48),
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    if st.button("Satz generieren"):
        st.session_state.llm_seed += 1
    rng = np.random.default_rng(st.session_state.llm_seed)
    satz = [startwort]
    wort = startwort
    for _ in range(14):
        kandidaten, probs_schritt = naechstes_wort_verteilung(wort, temperatur)
        if not kandidaten:
            break
        wort = rng.choice(kandidaten, p=probs_schritt)
        if wort == "■":
            break
        satz.append(wort)
    st.markdown(f"> **{' '.join(satz).capitalize()}.**")
    st.caption(
        "Jeder Klick auf „Satz generieren“ zieht eine neue Zufallssequenz. "
        "Bei hoher Temperature werden die Sätze kreativer, aber auch "
        "unsinniger."
    )
else:
    st.info("Wähle ein Startwort, um das Mini-Modell auszuprobieren.")

st.markdown(
    """
Nach demselben Prinzip arbeiten auch moderne LLMs, nur dass sie statt auf
das letzte Wort auf tausende Token Kontext zurückgreifen, statt zwanzig
Sätzen einen erheblichen Teil des Internets gelesen haben und statt einer
Zähltabelle ein tiefes **Transformer**-Netz verwenden, ein Neural Network
aus dem vorigen Kapitel, trainiert mit denselben Zutaten: Loss Function,
Gradient Descent, Milliarden Parameter. Der Qualitätssprung ist gewaltig,
der Kern jedoch bleibt derselbe: Gelernt wird, was in Texten worauf folgt.
"""
)

# ---------------------------------------------- Papagei oder Denker?
st.markdown("## Papagei oder Denker?")
st.markdown(
    """
Unser Korpus enthält zweimal das Muster *„der hahn kräht … die sonne geht
auf“*. Ein Sprachmodell lernt daraus, dass nach Hahn-Sätzen
Sonnenaufgangs-Wörter wahrscheinlich sind. Es kann sogar flüssig erklären,
dass Hähne die Sonne nicht verursachen, denn auch diese Erklärung steht
tausendfach in seinen Trainingstexten.

Vorsicht ist dennoch geboten mit dem Schluss, das Modell verstehe
Kausalität. Aus der Kausalitäts-Sektion kennst du die **Ladder of
Causation** (Judea Pearl):

1. **Association**: Was folgt worauf? Genau das leistet Next-Token-Training.
2. **Intervention**: Was geschieht, wenn wir eingreifen?
3. **Counterfactuals**: Was wäre gewesen, wenn?

Die Stufen 2 und 3 verlangen ein Modell der Welt, nicht nur ein Modell der
Texte über die Welt. Die Forschung streitet genau hierüber. Eine Position
bezeichnet LLMs als **Causal Parrots**: Sie rezitieren kausales Wissen aus
Texten, statt kausal zu schließen, und scheitern häufig an neuen, nie
beschriebenen Interventionsfragen. Die Gegenposition verweist auf
Benchmarks, in denen LLMs bei kausalen Aufgaben, etwa dem Bestimmen der
Kausalrichtung zwischen Variablen, erstaunlich gut abschneiden.
Möglicherweise reicht destilliertes Menschheitswissen für viele praktische
Zwecke bereits aus.
"""
)

merkkasten(
    "Merke",
    "LLMs lernen aus <b>Korrelationen in Texten</b>. Texte enthalten viel "
    "beschriebenes Kausalwissen, weshalb LLMs oft korrekt über Ursachen "
    "<i>reden</i>. Ob sie über neue Situationen hinweg kausal <i>schließen</i> "
    "können, ist eine offene Forschungsfrage und ein hervorragendes Thema "
    "für eigene Experimente.",
    typ="merke",
)

# ------------------------------------- LLMs als Werkzeug der Kausalanalyse
with vertiefung("LLMs als Werkzeug der Kausalanalyse"):
    st.markdown(
        """
    Die ergiebigere Frage ist oft nicht, ob das LLM kausal denken kann, sondern
    wofür es sich innerhalb einer Kausalanalyse einsetzen lässt. Drei
    Richtungen, die sich alle für Gruppenprojekte eignen:

    - **Causal Reasoning benchmarken:** Dem LLM systematisch Interventions- und
      Counterfactual-Fragen stellen, auch mit erfundenen Variablen, damit es
      nicht aus dem Gedächtnis antworten kann, und die Fehlerquote messen.
    - **Causal Discovery unterstützen:** Klassische Discovery-Algorithmen sehen
      nur Zahlen. LLMs kennen die Bedeutung von Variablennamen wie „Rauchen“
      oder „Lungenkrebs“ und können Kanten samt Richtung vorschlagen, die als
      Vorwissen in DAGs einfließen.
    - **Confounder aus Text gewinnen:** In unstrukturierten Daten wie
      Arztbriefen, Bewertungen oder Social-Media-Posts stecken Störfaktoren, die
      in keiner Tabellenspalte stehen. LLMs können sie extrahieren und messbar
      machen, sodass sich für sie adjustieren lässt.
    """
    )
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Kann ein LLM einen brauchbaren DAG vorschlagen? Das lässt sich "
            "messen: Gebt dem Modell Variablen aus einem Fachgebiet, "
            "vergleicht den vorgeschlagenen Graphen mit dem Expertenkonsens, "
            "und prüft, ob es die Richtung der Pfeile trifft."
        ),
        (
            "Wie unterscheidet man Auswendiglernen von Schlussfolgern? Wenn "
            "ein Modell ein Lehrbuchbeispiel richtig löst, kann es das "
            "Beispiel schlicht gelesen haben. Was ändert sich, wenn ihr Namen, "
            "Zahlen und Kontext austauscht?"
        ),
        (
            "Text als Confounder: Arztbriefe, Produktbewertungen und "
            "Gerichtsakten enthalten Störfaktoren, die in keiner "
            "Tabellenspalte stehen. Wie bringt man sie mit Embeddings in eine "
            "Kausalanalyse, und wann geht das schief?"
        ),
    ],
    hinweis=(
        "Startpunkt: die Benchmarks <code>Corr2Cause</code> und "
        "<code>CLADDER</code> für Causal Reasoning, dazu ein LLM eurer "
        "Wahl über API. Rechnet mit überraschend schwachen Ergebnissen, "
        "das ist ein Befund und kein Fehler."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- E. Kıcıman, R. Ness, A. Sharma & C. Tan (2023), *Causal Reasoning and Large Language Models: Opening a New Frontier for Causality*, arXiv:2305.00050
- M. Zečević, M. Willig, D. S. Dhami & K. Kersting (2023), *Causal Parrots: Large Language Models May Talk Causality But Are Not Causal*, Transactions on Machine Learning Research
- J. Pearl & D. Mackenzie (2018), *The Book of Why*, Basic Books (Ladder of Causation)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_dag, weiter_korr = st.columns(2)
with weiter_dag:
    st.page_link(
        "views/kausalitaet/dags_confounding.py",
        label="Das Handwerkszeug: DAGs & Confounding",
        icon="🕸️",
    )
with weiter_korr:
    st.page_link(
        "views/kausalitaet/korrelation.py",
        label="Grundlage: Korrelation ≠ Kausalität",
        icon="🔀",
    )
