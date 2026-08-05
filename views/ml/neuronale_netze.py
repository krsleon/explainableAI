"""Kapitel ML: Neural Networks.

Vom Neuron zum Netz, interaktives MLP-Training auf 2D-Daten, Brücke zu
Deep Learning und LLMs.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.ml_demos import entscheidungsgrenze, monde_daten
from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "🧠",
    "Neural Networks",
    "Vom einzelnen Neuron zur flexibelsten Modellfamilie des Machine Learning",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Ein künstliches **Neuron** ist mathematisch unspektakulär. Es bildet eine
gewichtete Summe seiner Eingaben und wendet darauf eine
**Activation Function** $\sigma$ an:

$$
\text{Ausgabe} = \sigma(w_1 x_1 + w_2 x_2 + \dots + w_k x_k + b).
$$

Ohne $\sigma$ bliebe dies eine lineare Funktion. Erst die Aktivierung, etwa
die **ReLU**, die negative Werte auf null setzt, führt die entscheidende
Nichtlinearität ein. Viele Neuronen nebeneinander bilden einen **Layer**,
mehrere Layer hintereinander ein tiefes Netz. Schon ein einziger,
ausreichend breiter Layer kann dabei nahezu beliebig komplexe Funktionen
zusammensetzen, dieses Resultat ist als *Universal Approximation Theorem*
bekannt.

Falls du das Kapitel **Lasso & Ridge** gelesen hast, kommt dir die Struktur
bekannt vor. Ein Netz mit einer verborgenen Schicht sagt vorher:

$$
\hat{g}(z) = \sum_{m=1}^{M} \beta_m \, \sigma(\alpha_m' z),
$$

das ist eine lineare Regression auf $M$ **konstruierten Regressoren**
$\sigma(\alpha_m'z)$! Der entscheidende Unterschied zum Wörterbuch $P(Z)$
von damals: Dort waren die Transformationen fest gewählt (Polynome,
Interaktionen), hier **lernt das Netz die Transformationen selbst**, denn
die inneren Gewichte $\alpha_m$ werden mitgeschätzt. Neural Networks sind
in diesem Sinn Regressionen, die sich ihr eigenes Wörterbuch aus den Daten
zusammenstellen, und bei mehreren Layern sogar Wörterbücher aus
Wörterbüchern.
"""
)

# ------------------------------------------------ Demo: MLP interaktiv
st.markdown("## Demo: Architektur und Decision Boundary")
st.markdown(
    """
In dieser Demo trainiert ein kleines **Multi-Layer Perceptron (MLP)** live
auf den Mond-Daten aus dem Trees-Kapitel. Wähle Architektur und Activation
Function und beobachte, welche Formen die Decision Boundary annehmen kann
und wie die **Loss Curve**, also der Fehler pro Trainingsepoche, verläuft.
"""
)

regler_schichten, regler_neuronen, regler_aktivierung = st.columns(3)
schichten = regler_schichten.slider("Hidden Layers", 1, 3, 1)
neuronen = regler_neuronen.select_slider(
    "Neuronen pro Layer", options=[2, 4, 8, 16, 32, 64], value=4
)
aktivierung = regler_aktivierung.selectbox(
    "Activation Function", ["relu", "tanh", "logistic"], index=0
)

X, y = monde_daten(n=400, rauschen=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


@st.cache_data
def netz_trainieren(schichten: int, neuronen: int, aktivierung: str, X_tr, y_tr):
    netz = MLPClassifier(
        hidden_layer_sizes=(neuronen,) * schichten,
        activation=aktivierung,
        max_iter=1500,
        random_state=0,
    )
    netz.fit(X_tr, y_tr)
    return netz


netz = netz_trainieren(schichten, neuronen, aktivierung, X_train, y_train)

spalte_grenze, spalte_verlust = st.columns(2)
with spalte_grenze:
    architektur = " × ".join([str(neuronen)] * schichten)
    st.plotly_chart(
        entscheidungsgrenze(netz, X_train, y_train, titel=f"MLP ({architektur}, {aktivierung})"),
        use_container_width=True,
    )
with spalte_verlust:
    fig_verlust = go.Figure()
    fig_verlust.add_scatter(
        y=netz.loss_curve_, mode="lines", name="Trainingsverlust",
        line=dict(color=FARBEN["beere"], width=3),
    )
    fig_verlust.update_layout(
        title="Loss Curve: Verlust pro Trainingsepoche",
        xaxis_title="Epoche", yaxis_title="Verlust", height=420,
    )
    st.plotly_chart(fig_verlust, use_container_width=True)

metrik_train, metrik_test = st.columns(2)
metrik_train.metric("Genauigkeit Training", f"{netz.score(X_train, y_train):.1%}")
metrik_test.metric("Genauigkeit Test", f"{netz.score(X_test, y_test):.1%}")

if schichten == 1 and neuronen <= 4:
    st.info(
        "**Geringe Kapazität:** Wenige Neuronen erzeugen nur wenige Knicke in "
        "der Decision Boundary, sie bleibt entsprechend grob (Underfitting). "
        "Erhöhe die Kapazität des Netzes."
    )
elif schichten >= 2 and neuronen >= 32:
    st.warning(
        "**Hohe Kapazität:** Das Netz kann die Monde vollständig umschließen "
        "und hat zugleich genug Spielraum, das Rauschen mitzulernen. Der "
        "Vergleich von Trainings- und Testgenauigkeit zeigt die aus dem "
        "ersten Kapitel bekannte Signatur des Overfittings."
    )

# ------------------------------------------ Wie lernt das Netz?
st.markdown("## Wie lernt das Netz? Loss, Gradient Descent, Backpropagation")
st.markdown(
    r"""
Woher kommen die Gewichte? Wie immer aus der Fehlerminimierung: Eine
**Loss Function** $L$ misst, wie weit die Vorhersagen von den Labels
entfernt sind (bei Regression der MSE, bei Klassifikation die
Cross-Entropy), und gesucht sind die Gewichte, die den Gesamtverlust
$\sum_i L\big(y_i, \hat{g}(z_i)\big)$ minimieren. Anders als bei der
linearen Regression gibt es dafür keine Formel, das Problem ist
**nichtkonvex**: Die Verlustlandschaft hat Täler, Sättel und lokale
Minima. Also tastet man sich iterativ hinab, mit **Gradient Descent**:

$$
w_{\text{neu}} = w_{\text{alt}} - \eta \cdot \nabla L(w_{\text{alt}}).
$$

Der **Gradient** $\nabla L$ zeigt in die Richtung des steilsten Anstiegs,
also geht man einen Schritt der Größe $\eta$ (**Lernrate**) in die
Gegenrichtung. Drei Vokabeln gehören dazu:

- **Backpropagation** ist der Algorithmus, der den Gradienten für alle
  Millionen Gewichte gleichzeitig effizient berechnet, im Kern die
  Kettenregel der Analysis, geschickt organisiert.
- **Stochastic Gradient Descent (SGD)** berechnet den Gradienten nicht auf
  allen Daten, sondern auf kleinen zufälligen **Mini-Batches**. Das ist
  billiger, und das Rauschen hilft sogar, aus schlechten lokalen Minima
  herauszuspringen. Moderne Optimizer wie **Adam** passen zusätzlich die
  Lernrate pro Gewicht automatisch an.
- Gegen Overfitting helfen dieselben Rezepte wie bisher: Strafterme auf
  die Gewichte (wie bei Ridge/Lasso) und **Early Stopping**, also
  aufhören, sobald der Fehler auf Validierungsdaten wieder steigt.
"""
)

st.markdown("### Demo: Gradient Descent auf einer Verlustlandschaft")
st.markdown(
    """
Die Kurve unten ist eine bewusst unbequeme Verlustlandschaft mit einem
tiefen Tal links und einem flacheren rechts. Wähle Startpunkt und Lernrate
und beobachte, wohin die Schritte führen.
"""
)

regler_start, regler_eta = st.columns(2)
startpunkt = regler_start.slider("Startpunkt w₀", -2.5, 2.5, 2.2, step=0.1)
eta = regler_eta.select_slider(
    "Lernrate η", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4], value=0.01
)


def verlust(w):
    return (w**2 - 1) ** 2 + 0.3 * w


def gradient(w):
    return 4 * w * (w**2 - 1) + 0.3


@st.cache_data
def gd_pfad(w0: float, eta: float, schritte: int = 40):
    pfad = [w0]
    w = w0
    for _ in range(schritte):
        w = w - eta * gradient(w)
        if abs(w) > 4:  # divergiert
            pfad.append(float(np.clip(w, -4, 4)))
            break
        pfad.append(w)
    return np.array(pfad)


pfad = gd_pfad(startpunkt, eta)
w_raster = np.linspace(-2.6, 2.6, 400)

fig_gd = go.Figure()
fig_gd.add_scatter(
    x=w_raster, y=verlust(w_raster), mode="lines", name="Verlustlandschaft L(w)",
    line=dict(color=FARBEN["schiefer"], width=2),
)
fig_gd.add_scatter(
    x=pfad, y=verlust(np.clip(pfad, -2.6, 2.6)), mode="lines+markers",
    name="Gradient-Descent-Pfad",
    line=dict(color=FARBEN["beere"], width=2),
    marker=dict(size=7),
)
fig_gd.add_scatter(
    x=[pfad[0]], y=[verlust(np.clip(pfad[0], -2.6, 2.6))], mode="markers",
    name="Start", marker=dict(color=FARBEN["nacht"], size=13, symbol="diamond"),
)
fig_gd.update_layout(
    xaxis_title="Gewicht w", yaxis_title="Verlust L(w)",
    yaxis=dict(range=[-1, 6]), height=420,
)
st.plotly_chart(fig_gd, use_container_width=True)

diverged = bool(np.abs(pfad[-1]) >= 4)
if diverged:
    st.warning(
        "**Divergenz:** Die Lernrate ist zu groß, jeder Schritt schießt über "
        "das Tal hinaus und der Verlust explodiert. Wähle ein kleineres η."
    )
elif eta <= 0.005:
    st.info(
        "**Zu vorsichtig:** Mit winziger Lernrate kriecht der Pfad und bleibt "
        "nach 40 Schritten weit vom Talboden entfernt. In der Praxis kostet "
        "das Rechenzeit und Geduld."
    )
elif pfad[-1] > 0:
    st.info(
        "**Lokales Minimum:** Der Pfad ist im rechten, flacheren Tal gelandet. "
        "Das tiefere Tal links bleibt unerreicht, ein Start auf der anderen "
        "Seite (oder das Rauschen von SGD) hätte es gefunden. Genau deshalb "
        "ist nichtkonvexe Optimierung schwer."
    )
else:
    st.success(
        "**Globales Minimum gefunden:** Der Pfad ist ins tiefere linke Tal "
        "gelaufen. Beachte, dass das vom Startpunkt abhing, nicht nur von der "
        "Lernrate."
    )

st.markdown(
    """
**Warum Tiefe?** Eine einzelne breite Schicht kann theoretisch jede Funktion
approximieren. Tiefe Netze bauen jedoch **Hierarchien** auf: Frühe Layer
lernen einfache Muster wie Kanten oder Silben, spätere kombinieren sie zu
abstrakten Konzepten wie Gesichtern oder Grammatik. Dass Deep Learning ab
etwa 2012 zum Durchbruch kam, verdankt es drei Zutaten, die gleichzeitig
zusammentrafen: **Daten** in nie gekannter Menge, **Hardware** (GPUs, die
tausende Gewichts-Updates parallel rechnen) und **Architekturen**, die
Vorwissen über die Datenart einbauen. **Convolutional Neural Networks
(CNNs)** etwa nutzen aus, dass in Bildern benachbarte Pixel
zusammengehören; rekurrente Netze und heute vor allem die
**Transformer**-Architektur verarbeiten Sequenzen wie Sprache. Auf
Transformern bauen die Large Language Models des nächsten Kapitels auf.
"""
)

merkkasten(
    "Merke",
    "Neural Networks sind außerordentlich flexible Funktionsapproximatoren: "
    "Nichtlineare Aktivierungen und gestapelte Layer erzeugen nahezu "
    "beliebige Formen. Der Preis dafür: Sie benötigen viele Daten, und "
    "niemand kann die Millionen Gewichte direkt <i>lesen</i>. Sie sind die "
    "namensgebende <b>Black Box</b> des Explainable-ML-Themas.",
    typ="merke",
)
gruppen_aufgabe(
    "Worauf eure Gruppe hier aufbaut",
    [
        (
            "Netze approximieren beliebige Funktionen. Was heißt das für ihre "
            "Rolle als Nuisance-Schätzer in Double Machine Learning, und woran "
            "scheitert die Sache trotzdem, wenn die Konvergenzrate nicht "
            "stimmt?"
        ),
        (
            "Wie erklärt man die Vorhersage eines Netzes? SHAP funktioniert "
            "technisch, aber die Erklärungen schwanken je nach "
            "Initialisierung. Prüft das nach, indem ihr dasselbe Netz mehrfach "
            "trainiert."
        ),
        (
            "Von hier zum Sprachmodell: Was kommt zwischen einem "
            "Feedforward-Netz und einem Transformer noch dazu? Das Kapitel "
            "LLMs und kausales Denken baut direkt darauf auf."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>pytorch</code> für eigene Architekturen, "
        "Goodfellow, Bengio und Courville, <i>Deep Learning</i> (frei "
        "online)."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- G. James, D. Witten, T. Hastie & R. Tibshirani (2021), *An Introduction to Statistical Learning*, 2. Aufl., Springer, Kap. 10 (frei online)
- I. Goodfellow, Y. Bengio & A. Courville (2016), *Deep Learning*, MIT Press, Kap. 6 (frei online)
- M. Nielsen (2015), *Neural Networks and Deep Learning*, Online-Buch (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_llm, weiter_xai = st.columns(2)
with weiter_llm:
    st.page_link(
        "views/ml/llms_kausalitaet.py",
        label="Weiter: LLMs & kausales Denken",
        icon="💬",
    )
with weiter_xai:
    st.page_link(
        "views/ml/explainable_ml.py",
        label="Oder: Explainable ML",
        icon="🔍",
    )
