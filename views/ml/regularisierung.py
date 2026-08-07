"""Kapitel ML: Lasso & Ridge.

Hochdimensionale Regression: Warum OLS bei vielen Regressoren versagt,
wie Ridge und Lasso mit Straftermen gegensteuern, Koeffizientenpfade, ein
Vorhersagevergleich auf dünn besetzten (sparsen) Daten und der
Regularisierungs-Bias als Preis dafür.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    lasso_path,
)

from utils.theming import (
    FARBEN,
    gruppen_aufgabe,
    kapitel_kopf,
    merkkasten,
    vertiefung,
)

kapitel_kopf(
    "🎚️",
    "Lasso & Ridge",
    "Viele Regressoren, wenige Beobachtungen: Regularisierung als eingebaute Selbstdisziplin",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Warum sollte man überhaupt mit sehr vielen Regressoren arbeiten? Aus zwei
Gründen. Erstens liefern moderne Datensätze schlicht **viele Merkmale**:
hunderte Eigenschaften einer Wohnung, eines Produkts, einer Patientin.
Zweitens, und das ist der subtilere Grund, kann man sich aus wenigen
Rohmerkmalen $Z$ beliebig viele **konstruierte Regressoren** bauen:
Potenzen, Interaktionen, Stufenfunktionen,

$$
X = P(Z) = \big(P_1(Z), \dots, P_p(Z)\big).
$$

Ein solches „Wörterbuch“ (*Dictionary*) von Transformationen kennst du
bereits: Die Polynomregression aus dem ersten Kapitel war genau das, mit
$P(Z) = (1, Z, Z^2, \dots)$. Je reicher das Wörterbuch, desto besser kann
die *lineare* Regression den **besten überhaupt möglichen Prädiktor**
approximieren, die bedingte Erwartung $g(Z) = E[Y \mid Z]$, und zwar auch
dann, wenn $g$ hochgradig nichtlinear ist.

Der Haken: Reiche Wörterbücher bedeuten großes $p$, und die **R²-Falle**
aus dem letzten Kapitel hat gezeigt, dass die klassische Regression dann
hemmungslos overfittet. Wir brauchen eine Regression mit eingebauter
Selbstdisziplin.
"""
)

# ---------------------------------------------- Abschnitt: Ridge & Lasso
st.markdown("## Die Idee: Ein Strafterm für große Koeffizienten")
st.markdown(
    r"""
**Ridge** und **Lasso** ergänzen das Kleinste-Quadrate-Kriterium um einen
**Strafterm** (*Penalty*), der große Koeffizienten teuer macht:

$$
\hat\beta_{\text{Ridge}} = \arg\min_b \;
\frac{1}{n}\sum_{i=1}^n \big(Y_i - b'X_i\big)^2
+ \lambda \sum_{j=1}^p b_j^2,
$$

$$
\hat\beta_{\text{Lasso}} = \arg\min_b \;
\frac{1}{n}\sum_{i=1}^n \big(Y_i - b'X_i\big)^2
+ \lambda \sum_{j=1}^p |b_j| .
$$

Der Tuning-Parameter $\lambda \geq 0$ steuert die Härte der Strafe: Bei
$\lambda = 0$ erhalten wir OLS zurück, bei sehr großem $\lambda$ werden
alle Koeffizienten gegen null gedrückt. Den besten Wert dazwischen wählt
man mit Data Splitting beziehungsweise **Cross-Validation**, also anhand
des Vorhersagefehlers auf Daten, die nicht zum Fitten benutzt wurden.

Der Unterschied zwischen den beiden wirkt winzig, Quadrat gegen Betrag, hat
aber eine bemerkenswerte Konsequenz:

- **Ridge** ($\ell_2$-Strafe) *schrumpft* alle Koeffizienten in Richtung
  null, setzt aber keinen einzigen exakt auf null.
- **Lasso** ($\ell_1$-Strafe) setzt unwichtige Koeffizienten **exakt auf
  null** und betreibt damit automatische Variablenselektion.

Die Intuition: Die Betragsfunktion hat bei null einen **Knick**. Um ihn zu
überwinden, muss ein Regressor einen echten Beitrag zur Fehlerreduktion
leisten, ein bisschen Rauschanpassung reicht nicht. Beim glatten Quadrat
der Ridge-Strafe spart der letzte Schritt auf die Null dagegen fast keine
Strafe mehr ein, kostet aber Anpassungsgüte, also unterbleibt er.
"""
)

merkkasten(
    "Definition",
    "<b>Regularisierung</b> heißt: Das Modell zahlt einen Preis für "
    "Komplexität und gibt dafür etwas In-sample-Fit auf, um out-of-sample "
    "besser vorherzusagen. Es tauscht also gezielt <b>Bias gegen Varianz</b>. "
    "Lasso straft Beträge und selektiert Variablen, Ridge straft Quadrate "
    "und schrumpft alle Koeffizienten.",
    typ="definition",
)

# --------------------------------------- Demo 1: Koeffizientenpfade
st.markdown("## Demo: Koeffizientenpfade")
st.markdown(
    r"""
Wir simulieren Daten mit $n = 120$ Beobachtungen und $p = 25$ Regressoren,
von denen nur **drei** wirklich zählen (wahre Koeffizienten $4$, $-3$ und
$2$, alle übrigen sind null). Die Abbildungen zeigen für jeden Regressor,
wie sein geschätzter Koeffizient von der Strafhärte $\lambda$ abhängt, die
farbigen Linien sind die drei echten Einflussgrößen, die grauen die 22
Störenfriede.
"""
)


@st.cache_data
def pfad_daten(n: int = 120, p: int = 25, seed: int = 7):
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:3] = [4.0, -3.0, 2.0]
    X = rng.standard_normal((n, p))
    y = X @ beta + rng.normal(0, 2.5, n)
    return X, y


@st.cache_data
def koeffizienten_pfade():
    X, y = pfad_daten()
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y)
    alphas_ridge = np.logspace(-2, 5, 60)
    coefs_ridge = np.array(
        [Ridge(alpha=a).fit(X, y).coef_ for a in alphas_ridge]
    ).T
    return alphas_lasso, coefs_lasso, alphas_ridge, coefs_ridge


def pfad_figur(alphas, coefs, titel: str) -> go.Figure:
    farben_echt = [FARBEN["gletscher"], FARBEN["sonne"], FARBEN["wiese"]]
    fig = go.Figure()
    for j in range(coefs.shape[0]):
        echt = j < 3
        fig.add_scatter(
            x=alphas,
            y=coefs[j],
            mode="lines",
            name=f"Regressor {j + 1}" if echt else "irrelevante Regressoren",
            line=dict(
                color=farben_echt[j] if echt else FARBEN["schiefer"],
                width=3 if echt else 1,
            ),
            opacity=1.0 if echt else 0.35,
            showlegend=echt or j == 3,
        )
    fig.update_layout(
        title=titel,
        xaxis_title="Strafhärte λ (log)",
        xaxis_type="log",
        yaxis_title="Geschätzter Koeffizient",
        height=420,
    )
    return fig


alphas_lasso, coefs_lasso, alphas_ridge, coefs_ridge = koeffizienten_pfade()

spalte_lasso, spalte_ridge = st.columns(2)
with spalte_lasso:
    st.plotly_chart(
        pfad_figur(alphas_lasso, coefs_lasso, "Lasso: Koeffizienten sterben exakt"),
        use_container_width=True,
    )
with spalte_ridge:
    st.plotly_chart(
        pfad_figur(alphas_ridge, coefs_ridge, "Ridge: alle schrumpfen, keiner stirbt"),
        use_container_width=True,
    )

st.markdown(
    r"""
Lies die Plots von rechts (harte Strafe) nach links (milde Strafe): Beim
**Lasso** tauchen mit sinkendem $\lambda$ zuerst die drei echten
Einflussgrößen auf, die grauen Störenfriede bleiben lange exakt bei null.
Bei **Ridge** sind dagegen bei jedem $\lambda$ alle 25 Koeffizienten von
null verschieden, sie werden nur gleichmäßig kleiner.

Dass das Lasso hier so zuverlässig die richtigen Variablen findet, liegt an
einer Annahme, die man **Approximate Sparsity** nennt: Wenige Regressoren
tragen viel bei, die Beiträge der übrigen sind vernachlässigbar klein
(zum Beispiel wie $\beta_j \propto 1/j^2$ abfallend). Für viele
ökonomische und wissenschaftliche Anwendungen ist das eine plausible
Beschreibung, und genau dann spielt das Lasso seine Stärke aus.
"""
)

# --------------------------------------- Demo 2: Vorhersagevergleich
st.markdown("## Demo: OLS, Ridge und Lasso im Vorhersage-Wettkampf")
st.markdown(
    r"""
Jetzt lassen wir die drei Verfahren gegeneinander antreten. Festes
$n = 100$, wachsendes $p$, Approximate Sparsity
($\beta_j = 3/j^2$), und gemessen wird der Fehler auf 2 000 frischen
Testbeobachtungen. Für Ridge und Lasso wird $\lambda$ per Cross-Validation
gewählt.
"""
)

if st.button("Neue Stichprobe", key="mse_neue_daten"):
    st.session_state["mse_seed"] = st.session_state.get("mse_seed", 1) + 1
mse_seed = st.session_state.get("mse_seed", 1)


@st.cache_data
def mse_vergleich(seed: int, n: int = 100):
    """Test-MSE von OLS, Ridge (CV) und Lasso (CV) über wachsendes p."""
    rng = np.random.default_rng(seed)
    p_werte = [5, 10, 20, 35, 50, 65, 80, 90, 100]
    p_max = max(p_werte)

    beta = 3.0 / (np.arange(1, p_max + 1) ** 2)
    X = rng.standard_normal((n, p_max))
    X_test = rng.standard_normal((2000, p_max))
    y = X @ beta + rng.normal(0, 2, n)
    y_test = X_test @ beta + rng.normal(0, 2, 2000)

    ergebnisse = {"OLS": [], "Ridge (CV)": [], "Lasso (CV)": []}
    for p in p_werte:
        Xp, Xp_test = X[:, :p], X_test[:, :p]
        ols = LinearRegression().fit(Xp, y)
        ridge = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(Xp, y)
        lasso = LassoCV(cv=5, random_state=0).fit(Xp, y)
        for name, modell in (
            ("OLS", ols), ("Ridge (CV)", ridge), ("Lasso (CV)", lasso)
        ):
            fehler = np.mean((y_test - modell.predict(Xp_test)) ** 2)
            ergebnisse[name].append(fehler)
    return p_werte, ergebnisse


p_werte, ergebnisse = mse_vergleich(mse_seed)

fig_mse = go.Figure()
for (name, fehler), farbe in zip(
    ergebnisse.items(), (FARBEN["beere"], FARBEN["gletscher"], FARBEN["wiese"])
):
    fig_mse.add_scatter(
        x=p_werte, y=fehler, mode="lines+markers", name=name,
        line=dict(color=farbe, width=3),
    )
fig_mse.add_hline(
    y=4.0, line_dash="dash", line_color=FARBEN["schiefer"],
    annotation_text="irreduzibles Rauschen (σ² = 4)",
)
fig_mse.update_layout(
    xaxis_title="Anzahl Regressoren p (bei n = 100)",
    yaxis_title="Test-MSE (log-Skala)",
    yaxis_type="log",
    height=440,
)
st.plotly_chart(fig_mse, use_container_width=True)

st.markdown(
    """
Solange $p$ klein ist, nehmen sich die drei wenig. Doch je näher $p$ an
$n$ rückt, desto dramatischer bricht OLS ein, bei $p = n$ interpoliert es
das Rauschen perfekt und versagt auf den Testdaten komplett. Ridge und
Lasso bleiben stabil nahe am irreduziblen Rauschen: Der Strafterm wirkt
wie eine Versicherung gegen Overfitting.

Das ist kein Kunstprodukt der Simulation. Auf realen US-Lohndaten mit 265
konstruierten Regressoren (Interaktionen von Alter, Ausbildung, Branche
usw. bei 1 500 Trainingsbeobachtungen) halbiert das Lasso in etwa den
Test-MSE gegenüber der ungestraften Regression und liegt gleichauf mit
deutlich aufwendigeren Verfahren wie Random Forests.
"""
)

# ------------------------------------------ Abschnitt: Regularisierungs-Bias
st.markdown("## Der Preis: Regularisierungs-Bias")
st.markdown(
    r"""
Regularisierung ist nicht umsonst. Der Strafterm zieht **jeden** Koeffizienten
in Richtung null, auch den, der dich eigentlich interessiert. Der Schätzer
landet deshalb im Mittel nicht auf dem wahren Wert, sondern systematisch zu
nah bei null. Diese Verzerrung heißt **Regularisierungs-Bias**, und sie ist
kein Unfall, sondern der Kaufpreis für die kleinere Varianz.

Sie verschwindet auch dann nicht, wenn du $\lambda$ per Cross-Validation
wählst. Im Gegenteil: CV sucht das $\lambda$ mit dem kleinsten
Vorhersagefehler, und dieses $\lambda$ ist gerade so groß, dass es etwas Bias
in Kauf nimmt, um viel Varianz zu sparen. Die Wahl ist optimal für die
Prognose und schief für die Schätzung.

Das lässt sich sichtbar machen. Wir halten das Design $X$ fest ($n = 100$,
$p = 80$), ziehen 200-mal neues Rauschen und schauen jedes Mal nur auf einen
einzigen Koeffizienten, den ersten, dessen wahrer Wert $\beta_1 = 1{,}5$
beträgt.
"""
)


@st.cache_data
def bias_varianz(n: int = 100, p: int = 80, wiederholungen: int = 200, seed: int = 11):
    """Sampling-Verteilung von beta_1 über Lambdas, bei festem Design X."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:3] = [1.5, 1.0, -0.8]
    X = rng.standard_normal((n, p))
    signal = X @ beta
    lambdas = np.logspace(-2.5, -0.2, 12)

    ols_schaetzer = np.empty(wiederholungen)
    lasso_schaetzer = np.empty((len(lambdas), wiederholungen))
    for r in range(wiederholungen):
        y = signal + rng.normal(0, 1.0, n)
        ols_schaetzer[r] = LinearRegression().fit(X, y).coef_[0]
        for k, lam in enumerate(lambdas):
            lasso_schaetzer[k, r] = Lasso(alpha=lam, max_iter=20000).fit(X, y).coef_[0]
    return beta[0], lambdas, ols_schaetzer, lasso_schaetzer


beta_wahr, lambdas, ols_schaetzer, lasso_schaetzer = bias_varianz()
bias = lasso_schaetzer.mean(axis=1) - beta_wahr
varianz = lasso_schaetzer.var(axis=1)
mse_beta = bias**2 + varianz
k_best = int(np.argmin(mse_beta))

fig_verteilung = go.Figure()
fig_verteilung.add_histogram(
    x=ols_schaetzer,
    name="OLS",
    marker_color=FARBEN["beere"],
    opacity=0.55,
    nbinsx=35,
)
fig_verteilung.add_histogram(
    x=lasso_schaetzer[k_best],
    name=f"Lasso (λ = {lambdas[k_best]:.2f})",
    marker_color=FARBEN["gletscher"],
    opacity=0.55,
    nbinsx=35,
)
fig_verteilung.add_vline(
    x=beta_wahr,
    line_dash="dash",
    line_color=FARBEN["schiefer"],
    annotation_text="wahrer Wert",
)
fig_verteilung.add_vline(
    x=float(lasso_schaetzer[k_best].mean()),
    line_dash="dot",
    line_color=FARBEN["gletscher"],
    annotation_text="Lasso im Mittel",
    annotation_position="bottom left",
)
fig_verteilung.update_layout(
    barmode="overlay",
    title="Sampling-Verteilung des ersten Koeffizienten",
    xaxis_title="Schätzung für β₁",
    yaxis_title="Häufigkeit",
    height=400,
)

fig_handel = go.Figure()
for werte, name, farbe in (
    (bias**2, "Bias²", FARBEN["sonne"]),
    (varianz, "Varianz", FARBEN["gletscher"]),
    (mse_beta, "MSE", FARBEN["beere"]),
):
    fig_handel.add_scatter(
        x=lambdas, y=werte, mode="lines+markers", name=name,
        line=dict(color=farbe, width=3),
    )
fig_handel.add_scatter(
    x=[lambdas[k_best]],
    y=[mse_beta[k_best]],
    mode="markers",
    name="bestes λ",
    marker=dict(color=FARBEN["nacht"], size=14, symbol="circle-open",
                line=dict(width=3)),
)
fig_handel.update_layout(
    title="Bias-Varianz-Zerlegung für β₁",
    xaxis_title="Strafhärte λ (log)",
    xaxis_type="log",
    yaxis_title="Beitrag zum MSE",
    height=400,
)

spalte_verteilung, spalte_handel = st.columns(2)
with spalte_verteilung:
    st.plotly_chart(fig_verteilung, use_container_width=True)
with spalte_handel:
    st.plotly_chart(fig_handel, use_container_width=True)

st.markdown(
    r"""
Links siehst du, wie sich die Schätzung über die 200 Stichproben verteilt.
OLS streut breit, liegt im Mittel aber richtig. Das Lasso streut deutlich
enger, sein Schwerpunkt liegt dafür sichtbar links vom wahren Wert. Dass eine
einzelne Lasso-Schätzung trotzdem über $1{,}5$ liegen kann, ändert nichts am
Befund: Der Fehler mittelt sich über Stichproben nicht weg, sondern bleibt.
Genau dieser Versatz ist der Regularisierungs-Bias.

Rechts steht der Handel im Detail. Mit wachsendem $\lambda$ fällt die Varianz
schnell, das quadrierte Bias wächst langsam, und der MSE hat sein Minimum
dazwischen. Das beste $\lambda$ ist also keines ohne Bias, sondern eines mit
gerade so viel Bias, dass die Varianzersparnis ihn aufwiegt.

Für die Vorhersage ist dieser Handel ein gutes Geschäft, für eine
Effektschätzung nicht. Steckt in $X$ eine Behandlungsvariable $D$, deren
Koeffizient dich als kausaler Effekt interessiert, dann schrumpft die Strafe
genau diesen Koeffizienten mit, und dein geschätzter Effekt fällt zu klein
aus. Dazu kommt der Selektionsschritt: Fliegt ein Confounder aus dem Modell,
weil sein Koeffizient knapp unter der Schwelle liegt, bleibt die Verzerrung
durch diesen Confounder unkorrigiert im Ergebnis stehen. Und die üblichen
Standardfehler gelten ohnehin nicht mehr, denn sie tun so, als wären
$\lambda$ und die Variablenauswahl vom Himmel gefallen statt aus denselben
Daten.

Der Ausweg besteht nicht darin, auf Regularisierung zu verzichten, ohne sie
wärst du in hohen Dimensionen chancenlos. Man regularisiert weiter, sorgt
aber dafür, dass der Bias den Zielparameter nicht mehr erreicht: Die
störenden Zusammenhänge schätzt ML, der Zielparameter wird dagegen
orthogonalisiert. Genau das leistet Double Machine Learning.
"""
)

with vertiefung("Warum genau um λ verschoben wird"):
    st.markdown(
        r"""
    Bei orthogonalem Design ($X'X/n = I$) lassen sich beide Schätzer in
    geschlossener Form hinschreiben. Ridge skaliert den OLS-Schätzer einfach
    herunter,

    $$
    \hat\beta_j^{\text{Ridge}} = \frac{\hat\beta_j^{\text{OLS}}}{1 + \lambda},
    $$

    das Lasso verschiebt ihn um einen festen Betrag Richtung null und schneidet
    ab, was darunter liegt (**Soft Thresholding**):

    $$
    \hat\beta_j^{\text{Lasso}} = \operatorname{sign}\big(\hat\beta_j^{\text{OLS}}\big)
    \cdot \max\Big(\big|\hat\beta_j^{\text{OLS}}\big| - \tfrac{\lambda}{2},\; 0\Big).
    $$

    Daran liest man beide Eigenschaften direkt ab: Ridge erreicht die Null nur
    im Grenzwert, das Lasso erreicht sie exakt, sobald der OLS-Wert kleiner als
    $\lambda/2$ ist.

    Wichtiger für uns ist die zweite Beobachtung. Der Bias hängt an $\lambda$,
    nicht an $n$. Er verschwindet nur, wenn $\lambda$ mit wachsender
    Stichprobe schnell genug gegen null geht, und dann verliert die Strafe
    ihre Schutzwirkung gegen Overfitting. Aus diesem Zielkonflikt kommst du
    durch eine klügere Wahl von $\lambda$ nicht heraus, sondern nur durch ein
    anderes Schätzverfahren.
    """
    )

merkkasten(
    "Achtung: selektiert heißt nicht ursächlich",
    "Das Lasso wählt Variablen, weil sie beim <b>Vorhersagen</b> helfen, "
    "nicht weil sie das Ergebnis <b>verursachen</b>. Von zwei hoch "
    "korrelierten Regressoren behält es oft willkürlich einen und verwirft "
    "den anderen, ein Confounder kann selektiert, die echte Ursache "
    "verworfen werden. Wer nach der Selektion naiv Koeffizienten "
    "interpretiert, begeht denselben Fehler wie bei der Feature Importance "
    "(Kapitel Trees &amp; Ensembles). "
    "Der saubere Ausweg (Orthogonalisierung, Double ML) kommt im Kapitel "
    "Kausales Machine Learning.",
    typ="achtung",
)
gruppen_aufgabe(
    "Worauf eure Gruppe hier aufbaut",
    [
        (
            "Lasso wählt Variablen aus. Darf man die ausgewählten Variablen "
            "anschließend kausal interpretieren? Die Antwort lautet nein, und "
            "der Grund dafür heißt Post-Selection Inference."
        ),
        (
            "Wie wählt ihr den Strafparameter, und wie stabil ist die Auswahl? "
            "Zieht mehrere Bootstrap-Stichproben und schaut, ob immer "
            "dieselben Variablen überleben."
        ),
        (
            "Ridge entspricht einem Gaussian Prior, Lasso einem Laplace-Prior "
            "auf den Koeffizienten. Rechnet diese Verbindung für euer Beispiel "
            "nach, sie schlägt die Brücke zum Kapitel Bayesian Methods."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>scikit-learn</code> (LassoCV, RidgeCV) und "
        "<code>DoubleML</code> für gültige Inferenz nach der "
        "Variablenauswahl. Zum Nachlesen Belloni, Chernozhukov und Hansen "
        "(2014)."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- G. James, D. Witten, T. Hastie & R. Tibshirani (2021), *An Introduction to Statistical Learning*, 2. Aufl., Springer, Kap. 6 (frei online)
- T. Hastie, R. Tibshirani & J. Friedman (2009), *The Elements of Statistical Learning*, 2. Aufl., Springer, Kap. 3 (frei online)
- V. Chernozhukov, C. Hansen, N. Kallus, M. Spindler & V. Syrgkanis (2024), *Applied Causal Inference Powered by ML and AI*, Kap. 3 (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    """
Ridge und Lasso bleiben lineare Modelle in einem fest gewählten Wörterbuch.
Die nächsten Kapitel gehen den anderen Weg: Modelle, die ihre
Nichtlinearität direkt aus den Daten lernen.
"""
)
weiter_trees, weiter_dml = st.columns(2)
with weiter_trees:
    st.page_link(
        "views/ml/baeume_ensembles.py", label="Weiter: Trees & Ensembles", icon="🌲"
    )
with weiter_dml:
    st.page_link(
        "views/kausalitaet/kausales_ml.py",
        label="Vorgriff: Kausales Machine Learning (DML)",
        icon="🎯",
    )
