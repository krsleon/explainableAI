"""Kapitel Kausalität: Bayesian Methods.

Prior → Daten → Posterior: Rechenregeln und Satz von Bayes, MLE vs. MAP,
Beta-Binomial-Modell, sequentielles Lernen, Bayesian Regression mit
Zerlegung in aleatorische und epistemische Unsicherheit sowie die Brücke
zu kausalen Fragen.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "🎲",
    "Bayesian Methods",
    "Wissen als Wahrscheinlichkeit: Vorwissen einbauen, mit Daten aktualisieren",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Bisher haben wir Wahrscheinlichkeit als **Häufigkeit** benutzt (wie oft
passiert etwas bei vielen Wiederholungen?). Die **bayesianische** Sichtweise
liest sie als **Grad der Überzeugung** und macht damit Sätze möglich wie:
*„Mit 90 % Wahrscheinlichkeit liegt die Wirksamkeit über 50 %.“*

Das Rezept ist immer dasselbe (Satz von Bayes):

$$
\underbrace{p(\theta \mid \text{Daten})}_{\text{Posterior}} \;\propto\;
\underbrace{p(\text{Daten} \mid \theta)}_{\text{Likelihood}} \times
\underbrace{p(\theta)}_{\text{Prior}}
$$

**Vorwissen** (Prior) trifft auf **Daten** (Likelihood) und wird zu
**aktualisiertem Wissen** (Posterior). Kein einzelner Schätzwert, sondern
eine ganze Verteilung dessen, was du glauben darfst.
"""
)

# ------------------------------------------- Motivation: Messbeispiel
st.markdown("## Motivation: eine Messung, viele mögliche Wahrheiten")
st.markdown(
    r"""
Ein Beispiel: Du misst morgens die Temperatur mit deinem Handy. Die Anzeige
sagt 8 °C, aber laut Datenblatt ist der Sensor ungenau, mit einer
Standardabweichung von 5 °C. Was war die wahre Temperatur $\mu$? Eine
einzelne Zahl wäre eine schlechte Antwort, denn sie würde die
Messunsicherheit unterschlagen. Ehrlicher ist eine **Verteilung** über
alle plausiblen Werte von $\mu$.

Dazu formulieren wir zwei Zutaten:

1. **Likelihood:** Wie entstehen Messwerte, falls die wahre Temperatur
   $\mu$ ist? Hier: $X \mid \mu \sim \mathcal{N}(\mu, 5^2)$.
2. **Prior:** Was hielten wir vor der Messung für plausibel? Etwa
   $\mu \sim \mathcal{N}(5, 10^2)$ für einen kühlen Morgen in den Bergen.

Der Satz von Bayes kombiniert beides zur Posterior-Verteilung
$p(\mu \mid x = 8)$. Kommt eine zweite Messung dazu, wird der Posterior
von eben zum neuen Prior, und das Update beginnt von vorn. Damit das kein
Rechentrick bleibt, schauen wir uns kurz an, warum es gar nicht anders
gehen kann.
"""
)

# ---------------------------------------- Theorie: Rechenregeln & Bayes
st.markdown("## Die Spielregeln der Wahrscheinlichkeit")
st.markdown(
    r"""
Die gesamte bayesianische Statistik folgt aus zwei Grundregeln. Für
Zufallsvariablen $X$ und $Y$ gilt:

$$
\underbrace{p(x) = \int p(x, y)\, dy}_{\text{Summenregel (Marginalisierung)}}
\qquad\qquad
\underbrace{p(x, y) = p(x \mid y)\, p(y)}_{\text{Produktregel}}
$$

Wendet man die Produktregel in beide Richtungen an,
$p(x, y) = p(x \mid y)\, p(y) = p(y \mid x)\, p(x)$, und stellt um, folgt
der **Satz von Bayes**. Für einen unbekannten Parameter $\theta$ und
beobachtete Daten $D$ lautet er vollständig:

$$
\underbrace{p(\theta \mid D)}_{\text{Posterior}}
= \frac{\overbrace{p(D \mid \theta)}^{\text{Likelihood}}\;
        \overbrace{p(\theta)}^{\text{Prior}}}{
  \underbrace{p(D)}_{\text{Evidenz}}},
\qquad
p(D) = \int p(D \mid \theta)\, p(\theta)\, d\theta.
$$

Die vier Bausteine im Einzelnen:

- **Likelihood** $p(D \mid \theta)$: unser explizites Modell dafür, wie
  die Daten entstanden sind, falls $\theta$ der wahre Parameter ist. Sind
  die Beobachtungen unabhängig, faktorisiert sie per Produktregel:
  $p(D \mid \theta) = \prod_{n=1}^N p(x_n \mid \theta)$. Wichtig für die
  Intuition: Bei festen Daten lesen wir die Likelihood als **Funktion von
  $\theta$**. Sie bewertet für jeden Kandidaten $\theta$, wie gut er die
  beobachteten Daten vorhergesagt hätte. Sie ist dabei keine Verteilung
  über $\theta$ (ihr Integral über $\theta$ ist im Allgemeinen nicht 1).
- **Prior** $p(\theta)$: unser Wissen über $\theta$, bevor wir die Daten
  sehen. Jede Analyse macht Annahmen. Der Prior macht sie explizit und
  damit kritisierbar.
- **Evidenz** $p(D)$ (auch Marginal Likelihood): die über den Prior
  gemittelte Likelihood. Sie normiert den Posterior und misst, wie gut
  das **Modell als Ganzes** die Daten erklärt, weshalb man mit ihr auch
  Modelle vergleichen kann. Da sie nicht von $\theta$ abhängt, darf man
  sie beim Rechnen oft weglassen. Genau daher kommt die Kurzform
  $\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$ von
  oben.
- **Posterior** $p(\theta \mid D)$: das Ergebnis. Eine vollständige
  Verteilung darüber, was wir nach den Daten über $\theta$ glauben
  dürfen.
"""
)

# ---------------------------------------------- Theorie: MLE, MAP, Prior
with vertiefung("Maximum Likelihood, MAP und der Prior als Regularisierung"):
    st.markdown(
        r"""
    Warum der Aufwand mit ganzen Verteilungen? Klassisches ML sucht meist nur
    den einen Parameterwert, der die Daten am wahrscheinlichsten macht
    (**Maximum Likelihood Estimation**, MLE):

    $$
    \hat\theta_{\text{MLE}}
    = \arg\max_\theta\; p(D \mid \theta)
    = \arg\max_\theta\; \sum_{n=1}^N \log p(x_n \mid \theta).
    $$

    Bei vielen Daten funktioniert das ausgezeichnet. Bei wenigen Daten wird
    MLE jedoch **überkonfident**. Heilt eine neue Therapie 3 von 3
    Patient:innen, liefert MLE $\hat\theta = 3/3 = 1$: die Behauptung, die
    Therapie wirke garantiert bei allen. Jede Unsicherheit ist aus der
    Antwort verschwunden, obwohl drei Beobachtungen kaum etwas belegen.

    Ein erster Schritt in Richtung Bayes ist, statt der Likelihood den
    Posterior zu maximieren (**Maximum a Posteriori**, MAP):

    $$
    \hat\theta_{\text{MAP}}
    = \arg\max_\theta\; \big[\log p(D \mid \theta) + \log p(\theta)\big].
    $$

    Der Log-Prior wirkt wie ein Strafterm, der unplausible Parameterwerte
    zurückhält. Für eine Regression mit Vorhersagefunktion $f^W$, Gaussian
    Noise $\varepsilon_n \sim \mathcal{N}(0, \sigma^2)$ und einem Prior
    $w_k \sim \mathcal{N}(0, s^2)$ auf den Gewichten wird daraus exakt die
    bekannte **Ridge-Regression** (L2-Regularisierung):

    $$
    \hat{W}_{\text{MAP}}
    = \arg\min_W\; \sum_{n=1}^N \big(y_n - f^W(x_n)\big)^2
    + \lambda\, \lVert W \rVert_2^2,
    \qquad \lambda = \frac{\sigma^2}{s^2}.
    $$

    Regularisierung ist also kein Trick, sondern ein Prior in Verkleidung.
    MAP liefert aber weiterhin nur einen Punkt. Erst der volle Posterior
    beantwortet Fragen wie „Mit welcher Wahrscheinlichkeit ist der Effekt
    größer als null?“. Wie das konkret aussieht, zeigt das folgende Beispiel.
    """
    )


# ------------------------------------------ Demo 1: Beta-Binomial
st.markdown("## Demo: Wie wirksam ist die Behandlung?")
st.markdown(
    r"""
Eine neue Therapie hat eine unbekannte Erfolgswahrscheinlichkeit $\theta$.
Als **Prior** wählen wir eine Beta-Verteilung, die Studiendaten ($k$ Erfolge
bei $n$ Behandelten) folgen einer Binomialverteilung. Diese Kombination ist
**konjugiert**: Der Posterior ist wieder eine Beta-Verteilung und lässt sich
geschlossen angeben:

$$
\theta \sim \mathrm{Beta}(\alpha, \beta), \quad
k \mid \theta \sim \mathrm{Bin}(n, \theta)
\;\;\Longrightarrow\;\;
\theta \mid k \sim \mathrm{Beta}(\alpha + k,\; \beta + n - k).
$$

Die Prior-Parameter lassen sich als „gedachte frühere Beobachtungen“ lesen:
$\alpha$ frühere Erfolge, $\beta$ frühere Misserfolge. Das Posterior-Mittel
ist ein **gewichtetes Mittel** aus Prior-Mittel und beobachteter
Erfolgsquote,

$$
E[\theta \mid k]
= \frac{\alpha + k}{\alpha + \beta + n}
= w\,\frac{\alpha}{\alpha + \beta} + (1 - w)\,\frac{k}{n},
\qquad w = \frac{\alpha + \beta}{\alpha + \beta + n},
$$

wobei das Gewicht $w$ des Vorwissens mit wachsendem $n$ gegen null geht.
"""
)

regler_prior, regler_daten = st.columns(2)
with regler_prior:
    st.markdown("**Prior: Was glaubst du vorher?**")
    alpha = st.slider("α (gedachte frühere Erfolge)", 0.5, 30.0, 1.0, step=0.5)
    beta = st.slider("β (gedachte frühere Misserfolge)", 0.5, 30.0, 1.0, step=0.5)
    st.caption(
        "α = β = 1: flach/ahnungslos · α klein, β groß: skeptisch · "
        "α groß, β klein: optimistisch"
    )
with regler_daten:
    st.markdown("**Daten: Was zeigt die Studie?**")
    n_patienten = st.slider("Patient:innen behandelt", 0, 200, 20)
    k_erfolge = st.slider("davon Erfolge", 0, max(n_patienten, 1), min(14, n_patienten))

theta = np.linspace(0.001, 0.999, 400)
prior = stats.beta.pdf(theta, alpha, beta)
posterior = stats.beta.pdf(theta, alpha + k_erfolge, beta + n_patienten - k_erfolge)

fig = go.Figure()
fig.add_scatter(
    x=theta, y=prior, mode="lines", name="Prior",
    line=dict(color=FARBEN["himmel"], width=2, dash="dash"),
)
if n_patienten > 0:
    likelihood = stats.binom.pmf(k_erfolge, n_patienten, theta)
    likelihood = likelihood / likelihood.max() * posterior.max() * 0.8
    fig.add_scatter(
        x=theta, y=likelihood, mode="lines", name="Likelihood (skaliert)",
        line=dict(color=FARBEN["sonne"], width=2, dash="dot"),
    )
fig.add_scatter(
    x=theta, y=posterior, mode="lines", name="Posterior",
    line=dict(color=FARBEN["nacht"], width=4), fill="tozeroy",
    fillcolor="rgba(30, 58, 95, 0.12)",
)
fig.update_layout(
    xaxis_title="Erfolgswahrscheinlichkeit θ", yaxis_title="Dichte", height=430,
)
st.plotly_chart(fig, use_container_width=True)

post = stats.beta(alpha + k_erfolge, beta + n_patienten - k_erfolge)
metrik_mittel, metrik_ki, metrik_p50 = st.columns(3)
metrik_mittel.metric("Posterior-Mittel", f"{post.mean():.1%}")
metrik_ki.metric(
    "95 %-Credible-Interval", f"{post.ppf(0.025):.0%} – {post.ppf(0.975):.0%}"
)
metrik_p50.metric("P(θ > 50 %)", f"{1 - post.cdf(0.5):.1%}")

st.markdown(
    """
Drei Beobachtungen lohnen das Ausprobieren. Erstens dominiert bei wenigen
Daten der Prior, während bei vielen Daten die Likelihood übernimmt, nahezu
unabhängig vom Vorwissen. Zweitens verlangt ein skeptischer Prior (etwa
α = 2, β = 8) mehr Evidenz, bevor er von der Therapie überzeugt ist, ganz
wie eine sorgfältige Gutachterin. Drittens darf man das **Credible
Interval** wörtlich nehmen: θ liegt mit 95 % Wahrscheinlichkeit in diesem
Bereich. Das ist genau der Satz, den viele beim frequentistischen
Konfidenzintervall aussprechen möchten, dort aber nicht dürfen.
"""
)

# ------------------------------------------ Demo 2: Sequentiell lernen
with vertiefung("Sequentielles Lernen mit dem Posterior als neuem Prior"):
    st.markdown(
        """
    Der Posterior von heute ist der Prior von morgen. Zieh den Regler und sieh
    zu, wie die Verteilung mit jedem Patientenschub wandert und schärfer wird
    (wahre Wirksamkeit in der Simulation: **70 %**).
    """
    )

    WAHRES_THETA = 0.7


    @st.cache_data
    def patienten_sequenz(n_max: int = 500, seed: int = 9):
        rng = np.random.default_rng(seed)
        return rng.binomial(1, WAHRES_THETA, n_max)


    gesehen = st.slider("Bisher behandelte Patient:innen", 0, 500, 0, step=10)
    ausgaenge = patienten_sequenz()[:gesehen]
    erfolge_seq = int(ausgaenge.sum())

    fig_seq = go.Figure()
    for m, deckkraft in [(0, 0.25), (gesehen // 2, 0.5), (gesehen, 1.0)]:
        if m == 0 and gesehen > 0:
            name = "Start (Prior, flach)"
        elif m == gesehen:
            name = f"nach {gesehen} Patient:innen"
        else:
            name = f"nach {m} Patient:innen"
        teil = patienten_sequenz()[:m]
        fig_seq.add_scatter(
            x=theta,
            y=stats.beta.pdf(theta, 1 + teil.sum(), 1 + m - teil.sum()),
            mode="lines", name=name, opacity=deckkraft,
            line=dict(color=FARBEN["nacht"], width=3),
        )
    fig_seq.add_vline(
        x=WAHRES_THETA, line_dash="dash", line_color=FARBEN["wiese"],
        annotation_text="wahres θ = 0,7",
    )
    fig_seq.update_layout(
        xaxis_title="Erfolgswahrscheinlichkeit θ", yaxis_title="Dichte", height=400,
    )
    st.plotly_chart(fig_seq, use_container_width=True)

    if gesehen > 0:
        st.caption(
            f"Stand: {erfolge_seq} Erfolge bei {gesehen} Behandelten "
            f"({erfolge_seq / gesehen:.0%})."
        )


# ------------------------------------- Theorie: Bayesian Regression
with vertiefung("Bayesian Regression, von Parametern zu Funktionen"):
    st.markdown(
        r"""
    Bisher war $\theta$ eine einzelne Zahl. Richtig interessant für Machine
    Learning wird die Sache, wenn wir dieselbe Logik auf **Funktionen**
    anwenden: Wir wollen eine Regressionsfunktion lernen und zugleich sagen
    können, wo wir ihr trauen dürfen.

    Dazu schreiben wir die Funktion als Linearkombination von $K$ festen
    Basisfunktionen $\phi_1, \dots, \phi_K$ (zum Beispiel Polynome oder
    Glockenkurven, in einem Neural Network wären es die gelernten Features
    der letzten Schicht):

    $$
    f^W(x) = \sum_{k=1}^{K} w_k\, \phi_k(x) = W^\top \phi(x).
    $$

    Unsere Annahmen halten wir als **Generative Story** fest, also als
    explizite Erzählung, wie die Daten entstanden sind:

    1. Die Natur wählt Gewichte $W$, unser Prior dafür ist
       $w_k \sim \mathcal{N}(0, s^2)$.
    2. Jede Beobachtung ist Funktionswert plus Rauschen:
       $y_n = f^W(x_n) + \varepsilon_n$ mit
       $\varepsilon_n \sim \mathcal{N}(0, \sigma^2)$.

    Ein Prior über Gewichte ist damit ein Prior über Funktionen: Ein kleines
    $s$ bevorzugt ruhige, flache Funktionen, ein großes $s$ erlaubt wilde
    Verläufe.

    Weil Prior und Likelihood beide Gaussian sind, ist das Modell wie beim
    Beta-Binomial-Beispiel **konjugiert**, und der Posterior über $W$ lässt
    sich geschlossen ausrechnen. Mit der Featurematrix
    $\Phi = [\phi(x_1), \dots, \phi(x_N)]^\top$ und
    $Y = (y_1, \dots, y_N)^\top$ gilt:

    $$
    W \mid D \sim \mathcal{N}(\mu', \Sigma'),
    \qquad
    \Sigma' = \big(\sigma^{-2}\, \Phi^\top \Phi + s^{-2} I_K\big)^{-1},
    \qquad
    \mu' = \sigma^{-2}\, \Sigma'\, \Phi^\top Y.
    $$

    Für die Vorhersage an einer neuen Stelle $x^*$ nehmen wir nicht das eine
    beste Modell, sondern **mitteln über alle plausiblen Gewichte**, gewichtet
    mit dem Posterior (Summen- und Produktregel):

    $$
    p(y^* \mid x^*, D)
    = \int p(y^* \mid x^*, W)\, p(W \mid D)\, dW
    = \mathcal{N}\Big(\mu'^\top \phi(x^*),\;
    \sigma^2 + \phi(x^*)^\top \Sigma'\, \phi(x^*)\Big).
    $$

    Diese **Predictive Distribution** ist das Herzstück: Ihr Mittelwert ist
    die Punktvorhersage, ihre Varianz sagt ehrlich, wie sicher sich das
    Modell an genau dieser Stelle ist.
    """
    )

    # --------------------------------- Demo 3: Regression mit Unsicherheit
    st.markdown("## Demo: Regression mit Unsicherheitsband")
    st.markdown(
        """
    Wir fitten die Bayesian Regression von oben mit Glockenkurven als
    Basisfunktionen an verrauschte Daten, die in der Mitte eine Lücke haben.
    Beobachte, wie sich das Band in der Lücke und an den Rändern aufbläht:
    Dort hat das Modell nichts gelernt und sagt das auch.
    """
    )

    regler_n_reg, regler_rausch, regler_s = st.columns(3)
    with regler_n_reg:
        n_reg = st.slider("Datenpunkte", 5, 100, 30, step=5)
    with regler_rausch:
        sigma_reg = st.slider("Rauschen σ", 0.05, 1.0, 0.3, step=0.05)
    with regler_s:
        s_reg = st.slider("Prior-Streuung s", 0.1, 5.0, 1.0, step=0.1)
    zeige_draws = st.checkbox(
        "Einzelne Funktionen aus dem Posterior einzeichnen", value=True
    )

    ZENTREN = np.linspace(0, 10, 14)
    RBF_BREITE = 0.9


    def merkmale(x: np.ndarray) -> np.ndarray:
        return np.exp(-((x[:, None] - ZENTREN[None, :]) ** 2) / (2 * RBF_BREITE**2))


    @st.cache_data
    def regressionsdaten(n: int, sigma: float, seed: int = 4):
        rng = np.random.default_rng(seed)
        links = rng.uniform(0.3, 3.8, n // 2)
        rechts = rng.uniform(6.2, 9.2, n - n // 2)
        x = np.sort(np.concatenate([links, rechts]))
        y = np.sin(x) + rng.normal(0, sigma, x.size)
        return x, y


    x_dat, y_dat = regressionsdaten(n_reg, sigma_reg)
    Phi = merkmale(x_dat)
    K_basis = ZENTREN.size
    Sigma_post = np.linalg.inv(
        Phi.T @ Phi / sigma_reg**2 + np.eye(K_basis) / s_reg**2
    )
    mu_post = Sigma_post @ Phi.T @ y_dat / sigma_reg**2

    x_stern = np.linspace(-1, 11, 300)
    Phi_stern = merkmale(x_stern)
    vorhersage = Phi_stern @ mu_post
    var_epistemisch = np.einsum("nk,kl,nl->n", Phi_stern, Sigma_post, Phi_stern)
    sd_gesamt = np.sqrt(sigma_reg**2 + var_epistemisch)
    sd_epistemisch = np.sqrt(var_epistemisch)

    fig_reg = go.Figure()
    fig_reg.add_scatter(
        x=x_stern, y=vorhersage + 1.96 * sd_gesamt, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
    fig_reg.add_scatter(
        x=x_stern, y=vorhersage - 1.96 * sd_gesamt, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(127, 168, 217, 0.25)",
        name="95 % gesamt (aleatorisch + epistemisch)", hoverinfo="skip",
    )
    fig_reg.add_scatter(
        x=x_stern, y=vorhersage + 1.96 * sd_epistemisch, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
    fig_reg.add_scatter(
        x=x_stern, y=vorhersage - 1.96 * sd_epistemisch, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(62, 109, 181, 0.3)",
        name="95 % nur epistemisch", hoverinfo="skip",
    )
    if zeige_draws:
        rng_draws = np.random.default_rng(1)
        cholesky_post = np.linalg.cholesky(
            Sigma_post + 1e-9 * np.eye(K_basis)
        )
        for i in range(5):
            w_draw = mu_post + cholesky_post @ rng_draws.standard_normal(K_basis)
            fig_reg.add_scatter(
                x=x_stern, y=Phi_stern @ w_draw, mode="lines",
                line=dict(color=FARBEN["flieder"], width=1), opacity=0.6,
                name="Funktionen aus dem Posterior", showlegend=(i == 0),
                hoverinfo="skip",
            )
    fig_reg.add_scatter(
        x=x_stern, y=np.sin(x_stern), mode="lines",
        line=dict(color=FARBEN["wiese"], width=2, dash="dash"),
        name="wahre Funktion sin(x)",
    )
    fig_reg.add_scatter(
        x=x_stern, y=vorhersage, mode="lines",
        line=dict(color=FARBEN["nacht"], width=3), name="Posterior-Mittel",
    )
    fig_reg.add_scatter(
        x=x_dat, y=y_dat, mode="markers",
        marker=dict(color=FARBEN["sonne"], size=7), name="Daten",
    )
    fig_reg.update_layout(xaxis_title="x", yaxis_title="y", height=460)
    st.plotly_chart(fig_reg, use_container_width=True)

    st.caption(
        "Vereinfachung: Wir tun so, als wäre das Rauschen σ bekannt. In der "
        "Praxis wird es mitgeschätzt. Die Lücke liegt zwischen x ≈ 4 und "
        "x ≈ 6, links von 0 und rechts von 9 extrapoliert das Modell."
    )


# ---------------------------------------- Zwei Arten von Unsicherheit
with vertiefung("Aleatorische und epistemische Unsicherheit"):
    st.markdown(
        r"""
    Die Varianz der Predictive Distribution zerfällt in zwei Teile mit sehr
    unterschiedlicher Bedeutung:

    $$
    \operatorname{Var}(y^* \mid x^*, D)
    = \underbrace{\sigma^2}_{\text{aleatorisch}}
    + \underbrace{\phi(x^*)^\top \Sigma'\, \phi(x^*)}_{\text{epistemisch}}.
    $$

    - **Aleatorische Unsicherheit** (von lateinisch *alea*, der Würfel):
      echtes Rauschen im Prozess, etwa Messfehler oder unvorhersehbare
      individuelle Schwankung. Sie kommt aus der Likelihood und bleibt
      bestehen, egal wie viele Daten wir sammeln. Würfeln wird nicht
      vorhersagbarer, wenn man öfter würfelt.
    - **Epistemische Unsicherheit** (von altgriechisch *episteme*, das
      Wissen): Unwissen über das Modell selbst. Sie kommt aus dem Posterior
      $\Sigma'$, ist groß an Stellen ohne Daten und schrumpft, sobald dort
      Beobachtungen dazukommen. In der Demo oben ist sie der Grund, warum
      sich das Band in der Datenlücke und an den Rändern aufbläht.

    Diese Zerlegung ist mehr als Buchhaltung. Sie beantwortet die praktische
    Frage, ob mehr Datensammeln hilft: gegen epistemische Unsicherheit ja,
    gegen aleatorische nein. Und sie warnt vor **Extrapolation**: Wo die
    epistemische Unsicherheit explodiert, verlässt das Modell den Bereich, in
    dem es etwas gelernt hat.
    """
    )


# ---------------------------------------------------- Bezug zur Kausalität
st.markdown("## Bezug zur Kausalinferenz")
st.markdown(
    """
Bayesianische Ideen tauchen in der Kausalinferenz überall auf:

- **Priors über Effektstärken:** Wundereffekte sind selten. Ein skeptischer
  Prior schützt vor der Überinterpretation kleiner Studien und ist die
  formale Version des Grundsatzes, dass außergewöhnliche Behauptungen
  außergewöhnliche Evidenz erfordern.
- **Bayesianische A/B-Tests und adaptive Studien:** Statt eines starren
  Stichprobenplans wird laufend die Posterior-Wahrscheinlichkeit berechnet,
  dass Variante A besser ist, und Studien dürfen früher gestoppt werden.
- **Sensitivity Analysis:** Wie stark müsste ein unbeobachteter Confounder
  sein, um den Effekt wegzuerklären? Priors machen solche
  Was-wäre-wenn-Fragen quantifizierbar.
- **Unsicherheit trennen und ausweisen:** Die Zerlegung in aleatorisch
  und epistemisch zeigt, wo ein Modell für Treatment-Effekte nur rät,
  etwa in Subgruppen ohne Beobachtungen oder außerhalb des Stützbereichs
  der Daten. Für Entscheidungen auf Basis geschätzter Effekte ist das oft
  wichtiger als die Punktschätzung selbst.
- **Werkzeuge:** `PyMC` (Python) und `Stan` schätzen beliebige
  Wahrscheinlichkeitsmodelle, einschließlich kompletter Kausal- und
  Mehrebenenmodelle.
"""
)

merkkasten(
    "Merke",
    "Bayes = Prior × Likelihood → Posterior. Vorwissen ist kein Schmuggel, "
    "sondern <b>explizit gemachte Annahme</b>, und bei wachsender Datenmenge "
    "verliert es automatisch an Gewicht. Credible Intervals sagen direkt, "
    "was man wissen will: die Wahrscheinlichkeit einer Hypothese gegeben "
    "die Daten. Vorhersage-Unsicherheit zerfällt in <b>aleatorisch</b> "
    "(Rauschen im Prozess, bleibt) und <b>epistemisch</b> (Unwissen über "
    "das Modell, schrumpft mit mehr Daten).",
    typ="merke",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Wie sehr hängt euer Ergebnis am Prior? Eine ehrliche "
            "bayesianische Analyse zeigt das, statt es zu verstecken: Prior "
            "Predictive Checks und Sensitivitätsanalyse über mehrere Priors."
        ),
        (
            "Was tut man, wenn es keine konjugierte Lösung gibt, also fast "
            "immer? Dann rechnet MCMC den Posterior numerisch aus. Woran "
            "erkennt man, dass der Sampler konvergiert ist, und woran, dass er "
            "es nicht ist?"
        ),
        (
            "Hierarchische Modelle: zwanzig Krankenhäuser, jedes mit wenigen "
            "Fällen. Getrennt schätzen ist zu verrauscht, zusammenwerfen "
            "verwischt die Unterschiede. Partial Pooling ist der Mittelweg. "
            "Wie stark zieht es die Schätzungen zusammen, und warum ist das "
            "keine Willkür?"
        ),
    ],
    hinweis=(
        "Startpunkt: <code>pymc</code> zum Modellieren, <code>arviz</code> "
        "für Diagnostik und Posterior-Plots. Als Begleitbuch R. McElreath, "
        "<i>Statistical Rethinking</i> (Vorlesungsvideos frei verfügbar)."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- R. McElreath (2020), *Statistical Rethinking*, 2. Aufl., CRC Press
- A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari & D. B. Rubin (2013), *Bayesian Data Analysis*, 3. Aufl., CRC Press (frei online)
- A. B. Downey (2021), *Think Bayes*, 2. Aufl., O'Reilly (frei online)
- C. M. Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, Kap. 3 (frei online)
- Y. Gal (2016), *Uncertainty in Deep Learning*, Dissertation, University of Cambridge (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_sem, weiter_po = st.columns(2)
with weiter_sem:
    st.page_link(
        "views/kausalitaet/sem_surveys.py",
        label="Weiter: SEMs & Survey Experiments",
        icon="📋",
    )
with weiter_po:
    st.page_link(
        "views/kausalitaet/potential_outcomes.py",
        label="Zurück: Potential Outcomes & RCTs",
        icon="⚖️",
    )
