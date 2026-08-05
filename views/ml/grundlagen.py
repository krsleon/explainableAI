"""Musterkapitel ML: Was ist Maschinelles Lernen?

Interaktive Einführung in Supervised Learning, Modellkomplexität und
Overfitting anhand einer Polynomregression.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "🤖",
    "Was ist Maschinelles Lernen?",
    "Lernen aus Beispielen, und warum Auswendiglernen nicht genügt",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Klassische Programme folgen Regeln, die ein Mensch explizit aufgeschrieben
hat. **Machine Learning (ML)** kehrt dieses Prinzip um: Wir geben dem
Computer **Beispiele** und lassen ihn die Regeln selbst finden.

Beim **Supervised Learning** besteht jedes Beispiel aus zwei Teilen:

- **Features** $x$: das, was wir beobachten (etwa Wohnfläche, Lage und Baujahr einer Wohnung)
- **Label** $y$: das, was wir vorhersagen wollen (etwa die Miete)

Gesucht ist eine Funktion $\hat{f}$ mit $\hat{f}(x) \approx y$, und zwar
ausdrücklich auch für **neue** Beobachtungen, die das Modell beim Lernen nie
gesehen hat. Dieser letzte Halbsatz ist der Kern des gesamten Kapitels.

Zwei Begriffspaare helfen, sich im ML-Dschungel zu orientieren. Erstens die
Art des Labels: Ist $y$ **stetig** (Miete, Umsatz, Temperatur), spricht man
von **Regression**; ist $y$ **kategorial** (Kredit ja/nein, Spam/kein
Spam), von **Classification**. Zweitens die Frage, ob es überhaupt ein
Label gibt: Unser Setting mit Label heißt **Supervised Learning**, das
Label „beaufsichtigt“ den Lernprozess. Beim **Unsupervised Learning**
beobachtet man nur Features und sucht Struktur in den Daten selbst, etwa
Gruppen ähnlicher Kundinnen (Clustering) oder komprimierte Darstellungen
(Dimensionsreduktion). Alle Kapitel dieser Sektion behandeln Supervised
Learning, beide Aufgabentypen kommen darin vor.
"""
)

merkkasten(
    "Definition",
    "<b>Machine Learning</b> bedeutet, aus Beispieldaten eine Funktion zu "
    "lernen, die auch auf <b>ungesehenen</b> Daten gute Vorhersagen liefert. "
    "Das Ziel ist Generalisierung, nicht Auswendiglernen.",
    typ="definition",
)

st.markdown(
    r"""
Formal wählt der Lernalgorithmus aus einer Modellklasse $\mathcal{F}$
diejenige Funktion, die den mittleren Fehler auf den Trainingsdaten
minimiert. Dieses Prinzip heißt **Empirical Risk Minimization**:

$$
\hat{f} = \arg\min_{f \in \mathcal{F}} \; \frac{1}{n} \sum_{i=1}^{n}
L\big(y_i, f(x_i)\big).
$$

Bei Regressionsproblemen verwendet man typischerweise den quadratischen
Verlust $L(y, \hat{y}) = (y - \hat{y})^2$, dessen Mittel der **Mean Squared
Error** (MSE) ist. Entscheidend ist am Ende jedoch nicht das empirische
Risiko auf den Trainingsdaten, sondern das erwartete Risiko auf **neuen**
Daten. Die Lücke zwischen beiden Größen ist das zentrale Thema dieses
Kapitels.
"""
)

# ------------------------------------------- Vorhersagen oder Verstehen?
st.markdown("## Vorhersagen oder Verstehen? Die zwei Fragen an ein Modell")
st.markdown(
    r"""
Bevor wir loslegen, lohnt eine Unterscheidung, die sich durch die gesamte
Website zieht. Mit einem gelernten $\hat{f}$ kann man zwei sehr
verschiedene Dinge wollen:

- **Prediction:** Wir brauchen nur gute Vorhersagen $\hat{f}(x)$ für neue
  Fälle. *Wie* das Modell rechnet, darf eine Black Box bleiben. Beispiel:
  Welchen Stundenlohn erwarten wir für eine Bewerberin mit gegebener
  Ausbildung und Erfahrung?
- **Inference:** Wir wollen die Rolle **einzelner Features** verstehen.
  Beispiel: Verdienen Frauen bei *gleicher* Ausbildung und Erfahrung
  weniger als Männer? Hier interessiert nicht die Vorhersage, sondern ein
  Parameter des Modells, und plötzlich zählt, was im Inneren von
  $\hat{f}$ vorgeht.

Machine Learning ist zunächst auf die erste Frage optimiert, und die
Kapitel dieser Sektion zeigen, wie weit man damit kommt. Die zweite Frage
erweist sich als heimtückischer: Sie führt über das Kapitel **Lineare
Regression** (wo sie eine präzise Antwort mit Kleingedrucktem bekommt)
geradewegs in die Kausalitäts-Sektion.
"""
)

# ---------------------------------------------- Demo 1: Polynomregression
st.markdown("## Demo: Lernen heißt Kurven anpassen")
st.markdown(
    """
Die Datenpunkte unten wurden von einem **wahren Zusammenhang** (gestrichelte
Linie) plus Zufallsrauschen erzeugt. In realen Daten kennen wir diese Linie
selbstverständlich nicht. Wir teilen die Punkte auf (**Data Splitting**):
in **Trainingsdaten**, an denen das Modell lernt, und **Testdaten**, die
das Modell nie zu sehen bekommt und die deshalb als ehrlicher Maßstab der
Generalisierung dienen.

Wähle den **Polynomgrad**, also die Flexibilität der gelernten Kurve, und
beobachte, wie sich Trainings- und Testfehler entwickeln.
"""
)


def wahre_funktion(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


@st.cache_data
def daten_erzeugen(n: int, rauschen: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = wahre_funktion(x) + rng.normal(0, rauschen, n)
    return x, y


if "ml_seed" not in st.session_state:
    st.session_state.ml_seed = 2026

regler_grad, regler_n, regler_rauschen, knopf = st.columns([2, 2, 2, 1])
grad = regler_grad.slider("Polynomgrad (Flexibilität)", 1, 15, 1)
n = regler_n.slider("Anzahl Datenpunkte", 20, 300, 60, step=20)
rauschen = regler_rauschen.slider("Rauschen", 0.05, 1.0, 0.3, step=0.05)
with knopf:
    st.markdown("&nbsp;")
    if st.button("Neue Daten"):
        st.session_state.ml_seed += 1

x, y = daten_erzeugen(n, rauschen, st.session_state.ml_seed)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)

modell = make_pipeline(PolynomialFeatures(grad), LinearRegression())
modell.fit(x_train.reshape(-1, 1), y_train)

raster = np.linspace(0, 1, 300)
vorhersage = modell.predict(raster.reshape(-1, 1))
mse_train = mean_squared_error(y_train, modell.predict(x_train.reshape(-1, 1)))
mse_test = mean_squared_error(y_test, modell.predict(x_test.reshape(-1, 1)))

fig = go.Figure()
fig.add_scatter(
    x=x_train, y=y_train, mode="markers", name="Trainingsdaten",
    marker=dict(color=FARBEN["gletscher"], size=8, opacity=0.8),
)
fig.add_scatter(
    x=x_test, y=y_test, mode="markers", name="Testdaten",
    marker=dict(color=FARBEN["sonne"], size=8, symbol="diamond", opacity=0.9),
)
fig.add_scatter(
    x=raster, y=wahre_funktion(raster), mode="lines", name="Wahrer Zusammenhang",
    line=dict(color=FARBEN["schiefer"], dash="dash", width=2),
)
fig.add_scatter(
    x=raster, y=vorhersage, mode="lines", name=f"Gelerntes Modell (Grad {grad})",
    line=dict(color=FARBEN["nacht"], width=3),
)
fig.update_layout(
    yaxis=dict(range=[-2.5, 2.5]), xaxis_title="Feature x", yaxis_title="Label y",
    height=440,
)
st.plotly_chart(fig, use_container_width=True)

metrik_train, metrik_test = st.columns(2)
metrik_train.metric("Fehler auf Trainingsdaten (MSE)", f"{mse_train:.3f}")
metrik_test.metric("Fehler auf Testdaten (MSE)", f"{mse_test:.3f}")

if grad <= 2:
    st.info(
        "**Underfitting:** Die Kurve ist zu starr, um den wahren Zusammenhang "
        "abzubilden, beide Fehler bleiben hoch. Versuche einen höheren Grad."
    )
elif grad >= 9:
    st.warning(
        "**Overfitting:** Die Kurve folgt jedem einzelnen Trainingspunkt und "
        "lernt damit das Rauschen mit. Der Trainingsfehler wird minimal, "
        "während der Testfehler deutlich steigt. Aufschlussreich ist auch die "
        "Schaltfläche „Neue Daten“: Die gelernte Kurve ändert sich von "
        "Stichprobe zu Stichprobe drastisch, obwohl der wahre Zusammenhang "
        "identisch bleibt. Das ist die hohe Varianz des Schätzers."
    )

# ------------------------------------- Demo 2: Fehler über alle Grade
st.markdown("## Trainings- und Testfehler über die Modellkomplexität")
st.markdown(
    """
Statt einzelne Grade durchzuprobieren, berechnen wir beide Fehler für alle
Grade von 1 bis 15. Es entsteht eine der wichtigsten Abbildungen des
Machine Learning: Der Trainingsfehler sinkt monoton, während der Testfehler
einen U-förmigen Verlauf nimmt. Links liegt Underfitting, rechts
Overfitting, und dazwischen der Bereich optimaler Komplexität.
"""
)


@st.cache_data
def fehlerkurven(x_tr, y_tr, x_te, y_te, max_grad: int = 15):
    grade = list(range(1, max_grad + 1))
    train_fehler, test_fehler = [], []
    for g in grade:
        m = make_pipeline(PolynomialFeatures(g), LinearRegression())
        m.fit(x_tr.reshape(-1, 1), y_tr)
        train_fehler.append(mean_squared_error(y_tr, m.predict(x_tr.reshape(-1, 1))))
        test_fehler.append(mean_squared_error(y_te, m.predict(x_te.reshape(-1, 1))))
    return grade, train_fehler, test_fehler


grade, train_fehler, test_fehler = fehlerkurven(x_train, y_train, x_test, y_test)
bester_grad = grade[int(np.argmin(test_fehler))]

fig_fehler = go.Figure()
fig_fehler.add_scatter(
    x=grade, y=train_fehler, mode="lines+markers", name="Trainingsfehler",
    line=dict(color=FARBEN["gletscher"], width=3),
)
fig_fehler.add_scatter(
    x=grade, y=test_fehler, mode="lines+markers", name="Testfehler",
    line=dict(color=FARBEN["sonne"], width=3),
)
fig_fehler.add_vline(
    x=grad, line_dash="dot", line_color=FARBEN["schiefer"],
    annotation_text="dein Grad", annotation_position="top",
)
fig_fehler.add_vline(
    x=bester_grad, line_dash="dash", line_color=FARBEN["wiese"],
    annotation_text="bester Testfehler", annotation_position="top right",
)
fig_fehler.update_layout(
    xaxis_title="Polynomgrad", yaxis_title="MSE (log-Skala)",
    yaxis_type="log", height=400,
)
st.plotly_chart(fig_fehler, use_container_width=True)

st.markdown(
    r"""
Hinter dem U-förmigen Verlauf steht die **Bias-Variance-Zerlegung** des
erwarteten Testfehlers an einer Stelle $x$:

$$
E\Big[\big(Y - \hat{f}(x)\big)^2\Big]
= \underbrace{\mathrm{Bias}\big[\hat{f}(x)\big]^2}_{\text{systematischer Fehler}}
+ \underbrace{\mathrm{Var}\big[\hat{f}(x)\big]}_{\text{Streuung über Stichproben}}
+ \underbrace{\sigma^2}_{\text{irreduzibles Rauschen}} .
$$

Mehr Flexibilität senkt den Bias und erhöht zugleich die Varianz, das
Minimum der Summe liegt bei mittlerer Komplexität. Der Rauschterm $\sigma^2$
bildet die Untergrenze, die kein noch so gutes Modell unterschreiten kann.

Die drei Terme kannst du in der Demo oben direkt beobachten: Der **Bias**
ist die systematische Abweichung der gelernten Kurve vom wahren
Zusammenhang (bei Grad 1 unübersehbar). Die **Varianz** zeigt die
Schaltfläche „Neue Daten“: Bei hohem Grad springt die gelernte Kurve von
Stichprobe zu Stichprobe wild umher, obwohl sich am wahren Zusammenhang
nichts ändert. Und selbst das perfekte Modell, die gestrichelte Linie,
trifft die einzelnen Punkte nicht, das ist $\sigma^2$. Derselbe Tradeoff
kehrt in jedem folgenden Kapitel in neuem Gewand wieder: als Baumtiefe,
als Strafhärte $\lambda$, als Netzwerkgröße.
"""
)

merkkasten(
    "Merke",
    "Ein Modell ist so gut wie sein Fehler auf <b>ungesehenen</b> Daten. "
    "Mehr Flexibilität senkt den Trainingsfehler immer, den Testfehler jedoch "
    "nur bis zu einem Punkt. Danach lernt das Modell Rauschen statt Struktur. "
    "Dieses Spannungsverhältnis heißt <b>Bias-Variance-Tradeoff</b>.",
    typ="merke",
)
gruppen_aufgabe(
    "Worauf eure Gruppe hier aufbaut",
    [
        (
            "Der Bias-Variance-Tradeoff taucht in jedem Verfahren dieser "
            "Website wieder auf, von der RDD-Bandbreite über die Stärke des "
            "Priors bis zur Tiefe eines Trees. Sucht ihn in eurem Gruppenthema "
            "und benennt konkret, was dort Bias und was Varianz ist."
        ),
        (
            "Wir wählen hier über einen einzigen Testfehler aus. Wie macht man "
            "das seriös? Kreuzvalidierung, Nested Cross-Validation, und die "
            "Frage, warum wiederholtes Nachschauen auf demselben Testset die "
            "Fehlerschätzung ruiniert."
        ),
        (
            "Vorhersage und Kausalität sind zwei verschiedene Ziele. Legt für "
            "euer Thema fest, welches der beiden ihr verfolgt, denn davon "
            "hängt ab, ob ein besseres Modell auch eine bessere Antwort "
            "liefert."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>scikit-learn</code> (Model Selection) und ISLR "
        "Kap. 5. Die Antwort auf die dritte Frage gehört in Abschnitt 1 "
        "eurer Projektarbeit."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- G. James, D. Witten, T. Hastie & R. Tibshirani (2021), *An Introduction to Statistical Learning*, 2. Aufl., Springer, Kap. 2 (frei online)
- T. Hastie, R. Tibshirani & J. Friedman (2009), *The Elements of Statistical Learning*, 2. Aufl., Springer, Kap. 2 und 7 (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    """
Polynome von Hand zu wählen ist erst der Anfang. Als Nächstes schauen wir
uns das Arbeitspferd des Feldes genauer an, die lineare Regression, und
stellen ihr beide Fragen: die nach der Vorhersage und die nach dem
Verstehen. Und am Ende der Sektion wartet die Einsicht, die dieses ganze
Projekt antreibt: Ein Modell, das hervorragend **vorhersagt**, weiß deshalb
noch lange nicht, **warum** etwas geschieht.
"""
)
weiter_ml, weiter_kausal = st.columns(2)
with weiter_ml:
    st.page_link(
        "views/ml/lineare_regression.py",
        label="Weiter: Lineare Regression",
        icon="📈",
    )
with weiter_kausal:
    st.page_link(
        "views/kausalitaet/korrelation.py",
        label="Oder direkt: Korrelation ≠ Kausalität",
        icon="🔀",
    )
