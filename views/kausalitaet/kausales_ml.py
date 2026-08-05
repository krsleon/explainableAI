"""Kapitel Kausalität: Kausales Machine Learning (DoubleML).

Warum naive ML-Schätzung kausaler Effekte scheitert und wie
Double/Debiased Machine Learning (Partialling-Out und Cross-Fitting) das
Problem löst, mit Simulationsvergleich.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "🎯",
    "Kausales Machine Learning",
    "ML-Flexibilität + kausale Garantien: Double/Debiased Machine Learning",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Angenommen, du willst den Effekt eines Treatments $D$ auf ein Outcome $Y$
schätzen und verfügst über viele Kontrollvariablen $X$, womöglich hunderte
Spalten mit unbekannten Nichtlinearitäten. Die klassische lineare Regression
zwingt dich, die funktionale Form der Zusammenhänge zu raten. ML-Modelle
könnten diese Zusammenhänge flexibel lernen.

Allerdings ist ML auf **Prediction** ausgelegt, nicht auf Effektschätzung.
Ein Random Forest besitzt keinen Koeffizienten für $D$, und die
Regularisierung, die Overfitting zähmt, verzerrt systematisch alles, was man
an Effekten herauslesen möchte. Der Ausweg heißt **Double/Debiased Machine
Learning (DML)**, und sein Kern ist eine bemerkenswert alte Idee.
"""
)

st.markdown("## Das Modell: Partially Linear Regression")
st.markdown(
    r"""
DML wird üblicherweise im **partiell linearen Modell** (Robinson 1988)
formuliert:

$$
Y = \tau\, D + g(X) + \zeta, \qquad E[\zeta \mid D, X] = 0,
$$

$$
D = m(X) + \nu, \qquad E[\nu \mid X] = 0.
$$

Der Zielparameter ist $\tau$. Die **Nuisance-Funktionen** $g(X)$ (Einfluss
der Confounder auf das Outcome) und $m(X)$ (Treatment-Gleichung) sind
unbekannt und dürfen beliebig nichtlinear sein. Genau hier kommt ML ins
Spiel.

Der Schlüssel ist eine klassische Einsicht (**Frisch–Waugh–Lovell**): Der
Effekt von $D$ auf $Y$ unter Kontrolle von $X$ steckt vollständig in den
**Residuen**:

1. $\tilde{Y} = Y - \hat{E}[Y \mid X]$: das Outcome, um den $X$-Anteil bereinigt
2. $\tilde{D} = D - \hat{E}[D \mid X]$: das Treatment, um den $X$-Anteil bereinigt
3. Die Regression von $\tilde{Y}$ auf $\tilde{D}$ liefert $\hat{\tau}$.

DML ersetzt die bedingten Erwartungswerte in Schritt 1–2 durch beliebige
ML-Verfahren (Random Forest, Boosting, neuronale Netze). Damit das gutgeht,
braucht es zwei Sicherungen: **Cross-Fitting** und **Neyman-Orthogonalität**.
"""
)

with vertiefung("Die zwei Sicherungen: Cross-Fitting und Orthogonalität"):
    st.markdown(
        r"""
    **Cross-Fitting.** Die Vorhersage für jede Beobachtung stammt von einem
    Modell, das diese Beobachtung nie im Training gesehen hat. Andernfalls
    überträgt sich Overfitting auf die Residuen: Der Wald sagt die
    Trainingsdaten zu gut vorher, $\tilde{Y}$ und $\tilde{D}$ werden zu klein,
    und der Effekt verzerrt sich systematisch. Praktisch teilt man die
    Stichprobe in $K$ Folds, trainiert auf $K-1$ und sagt den ausgelassenen
    vorher.

    **Neyman-Orthogonalität.** Die Momentenbedingung

    $$
    E\big[(\tilde{Y} - \tau\,\tilde{D})\,\tilde{D}\big] = 0
    $$

    ist so konstruiert, dass ihre Ableitung nach den Nuisance-Funktionen im
    wahren Parameter verschwindet. Schätzfehler in $\hat{g}$ und $\hat{m}$
    wirken sich deshalb nur in **zweiter Ordnung** auf $\hat{\tau}$ aus. ML
    darf also unpräzise sein, solange beide Nuisance-Schätzungen hinreichend
    schnell konvergieren. Die übliche Bedingung ist eine Rate von
    $o(n^{-1/4})$ für jede der beiden.
    """
    )

# ------------------------------------------------ Demo: DML vs. naiv
st.markdown("## Demo: Drei Schätzer im Vergleich")
st.markdown(
    """
Simulation mit **nichtlinearem Confounding**: Fünf Kontrollvariablen
beeinflussen (über Sinus- und Quadrat-Terme) sowohl die Behandlung $D$ als
auch das Outcome $Y$. Der wahre Effekt $\\tau$ ist dir bekannt. Welcher
Estimator findet ihn?

1. **Naiv:** Regression $Y \\sim D$, die $X$ vollständig ignoriert
2. **Linear:** Regression $Y \\sim D + X$, die kontrolliert, aber nur linear
3. **DML:** Partialling-Out mit Random Forests und Cross-Fitting
"""
)

regler_tau, regler_gamma, knopf = st.columns([2, 2, 1])
tau = regler_tau.slider("Wahrer Effekt τ", 0.0, 2.0, 1.0, step=0.1)
gamma = regler_gamma.slider("Stärke des Confoundings γ", 0.0, 2.0, 1.0, step=0.1)
with knopf:
    st.markdown("&nbsp;")
    if st.button("🎲 Neue Stichprobe"):
        st.session_state["dml_seed"] = st.session_state.get("dml_seed", 0) + 1
seed = st.session_state.get("dml_seed", 0)


@st.cache_data
def dml_vergleich(tau: float, gamma: float, seed: int, n: int = 1200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 5))
    stoerfunktion = 2 * np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + X[:, 2]
    d = gamma * stoerfunktion + rng.normal(0, 1, n)
    y = tau * d + gamma * stoerfunktion + rng.normal(0, 1, n)

    # 1) Naiv: y ~ d
    naiv = np.polyfit(d, y, 1)[0]

    # 2) Linear: y ~ d + X
    design = np.column_stack([d, X, np.ones(n)])
    linear = np.linalg.lstsq(design, y, rcond=None)[0][0]

    # 3) DML: Cross-Fitting mit 2 Folds, RF für beide Nuisance-Funktionen
    haelfte = n // 2
    folds = [(np.arange(haelfte), np.arange(haelfte, n)),
             (np.arange(haelfte, n), np.arange(haelfte))]
    rest_y, rest_d = np.zeros(n), np.zeros(n)
    for train, test in folds:
        rf_y = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0)
        rf_d = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0)
        rf_y.fit(X[train], y[train])
        rf_d.fit(X[train], d[train])
        rest_y[test] = y[test] - rf_y.predict(X[test])
        rest_d[test] = d[test] - rf_d.predict(X[test])
    dml = np.polyfit(rest_d, rest_y, 1)[0]

    return naiv, linear, dml


naiv, linear, dml = dml_vergleich(tau, gamma, seed)

fig = go.Figure()
fig.add_bar(
    x=["Naiv (Y ~ D)", "Linear (Y ~ D + X)", "DML (RF + Cross-Fitting)"],
    y=[naiv, linear, dml],
    marker_color=[FARBEN["beere"], FARBEN["sonne"], FARBEN["wiese"]],
    text=[f"{naiv:.2f}", f"{linear:.2f}", f"{dml:.2f}"],
    textposition="outside",
)
fig.add_hline(
    y=tau, line_dash="dash", line_color=FARBEN["schiefer"],
    annotation_text=f"wahrer Effekt τ = {tau:.1f}",
)
fig.update_layout(yaxis_title="geschätzter Effekt", height=420)
st.plotly_chart(fig, use_container_width=True)

if gamma > 0.2:
    st.markdown(
        f"""
Der **naive** Estimator ({naiv:.2f}) enthält das vollständige Confounding.
Der **lineare** ({linear:.2f}) beseitigt nur dessen linearen Anteil, die
Sinus- und Quadrat-Komponenten verbleiben als Restverzerrung. **DML**
({dml:.2f}) approximiert die Nuisance-Funktionen flexibel und liegt nahe am
wahren τ = {tau:.1f}. Wiederholte Ziehungen über „Neue Stichprobe“ zeigen
zusätzlich die Stichprobenstreuung der drei Estimatoren.
"""
    )
else:
    st.info(
        "Bei γ ≈ 0 gibt es kein Confounding, alle drei Estimatoren treffen. "
        "Erhöhe γ, um die Unterschiede sichtbar zu machen."
    )

merkkasten(
    "Warum Cross-Fitting?",
    "Würde der Random Forest auf denselben Daten vorhersagen, auf denen er "
    "trainiert wurde, wären die Residuen zu klein und systematisch verzerrt, "
    "Overfitting sickert in die Effektschätzung. Cross-Fitting trennt "
    "Trainieren und Vorhersagen strikt: Jede Beobachtung wird von einem "
    "Modell bewertet, das sie nie gesehen hat.",
    typ="definition",
)

st.markdown(
    """
**Und darüber hinaus:** DML schätzt nicht nur Durchschnittseffekte.
Verwandte Verfahren wie **Causal Forests** und der **DR-Learner** schätzen
**Heterogeneous Treatment Effects**: Für wen wirkt eine Maßnahme stark, für
wen gar nicht? Das ist die methodische Grundlage personalisierter Medizin
und zielgenauer Politikgestaltung.

**Praxis-Hinweis:** Das Python- und R-Paket **`DoubleML`** implementiert das
Framework inklusive gültiger Inferenz mit Standardfehlern und
Konfidenzintervallen. Es wurde maßgeblich am Lehrstuhl für Statistik der
**Universität Hamburg** mitentwickelt, von dem auch diese
Akademie-Arbeitsgruppe kommt.
"""
)

merkkasten(
    "Merke",
    "DML = Partialling-Out mit ML für die Störfunktionen + Cross-Fitting + "
    "Orthogonalität. <b>ML macht die Vorhersage-Arbeit, die Ökonometrie "
    "liefert die kausale Garantie.</b> Aber: DML adjustiert nur für "
    "<i>beobachtete</i> Confounder. Welche das sein müssen, sagt dir weiterhin "
    "der DAG, nicht der Algorithmus.",
    typ="merke",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Für <i>wen</i> wirkt eine Maßnahme? Der ATE ist ein Durchschnitt "
            "und kann gegensätzliche Einzeleffekte verdecken. Causal Forests "
            "und der DR-Learner schätzen heterogene Effekte. Wie verhindert "
            "man dabei, dass man nur Rauschen findet?"
        ),
        (
            "Wie gut müssen die ML-Nuisance-Modelle sein? Die Theorie verlangt "
            "eine Konvergenzrate, die man an echten Daten nie nachprüfen kann. "
            "Welche praktischen Diagnosen gibt es ersatzweise?"
        ),
        (
            "DML adjustiert nur für <i>beobachtete</i> Confounder. Wie stark "
            "müsste ein unbeobachteter Confounder sein, um euer Ergebnis zu "
            "kippen? Genau das beantwortet eine Sensitivitätsanalyse."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>DoubleML</code> (mitentwickelt am Lehrstuhl für "
        "Statistik der Universität Hamburg) und <code>econml</code> für "
        "CATE-Schätzung. Dazu Chernozhukov et al., <i>Applied Causal "
        "Inference Powered by ML and AI</i> (frei online)."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey & J. Robins (2018), *Double/Debiased Machine Learning for Treatment and Structural Parameters*, The Econometrics Journal 21(1), C1–C68
- P. Bach, V. Chernozhukov, M. S. Kurz & M. Spindler (2022), *DoubleML: An Object-Oriented Implementation of Double Machine Learning in Python*, Journal of Machine Learning Research 23(53), 1–6
- M. Huber (2023), *Causal Analysis: Impact Evaluation and Causal Machine Learning with Applications in R*, MIT Press
"""
)

st.markdown("## Wie geht es weiter?")
weiter_bayes, weiter_xai = st.columns(2)
with weiter_bayes:
    st.page_link(
        "views/kausalitaet/bayes.py", label="Weiter: Bayesian Methods", icon="🎲"
    )
with weiter_xai:
    st.page_link(
        "views/ml/explainable_ml.py",
        label="Passt dazu: Explainable ML",
        icon="🔍",
    )
