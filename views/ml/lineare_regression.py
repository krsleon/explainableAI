"""Kapitel ML: Lineare Regression.

Die zwei Fragen der Regressionsanalyse (Prediction vs. Inference), das
Best-Linear-Prediction-Problem, die R²-Falle bei vielen Regressoren und
Partialling-Out (Frisch-Waugh-Lovell) am Beispiel Gender Wage Gap.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "📈",
    "Lineare Regression",
    "Zwei Fragen an dieselben Daten: Wie gut können wir vorhersagen, und was bedeutet ein Koeffizient?",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Die lineare Regression ist über 200 Jahre alt und trotzdem das Arbeitspferd
des modernen Machine Learning: Fast alle Methoden dieser Sektion, von Lasso
bis zu Neural Networks, sind Antworten auf Schwächen der linearen
Regression und werden erst vor ihrem Hintergrund verständlich.

Ausgangspunkt ist ein Beispiel, das uns durch mehrere Kapitel begleiten
wird. $Y$ sei der **Stundenlohn** einer Person, $X$ ein Vektor von
Merkmalen: Ausbildung, Berufserfahrung, Region, Geschlecht. An diese Daten
lassen sich zwei grundverschiedene Fragen stellen:

1. **Die Prediction Question:** Wie nutzen wir $X$, um $Y$ möglichst gut
   vorherzusagen? *(Welchen Lohn erwarten wir für eine Person mit diesen
   Merkmalen?)*
2. **Die Inference Question:** Wie ändert sich der vorhergesagte Lohn, wenn
   wir **ein** Merkmal $D$ ändern und alle übrigen Merkmale $W$ festhalten?
   *(Verdienen Frauen bei gleicher Ausbildung und Erfahrung weniger als
   Männer?)*

Für die zweite Frage zerlegen wir den Regressorvektor in $X = (D, W)$: das
**Zielmerkmal** $D$, dessen Rolle uns interessiert, und die **Kontrollen**
$W$. Diese Unterscheidung wirkt hier unscheinbar, sie ist aber der rote
Faden, der die ML-Kapitel mit der gesamten Kausalitäts-Sektion verbindet.
"""
)

merkkasten(
    "Definition",
    "<b>Prediction</b> fragt nach guten Vorhersagen und behandelt das Modell "
    "als Black Box. <b>Inference</b> fragt nach der Rolle einzelner Merkmale "
    "und damit nach dem Modell selbst. Dieselbe Regression beantwortet beide "
    "Fragen, aber mit sehr unterschiedlichen Anforderungen.",
    typ="definition",
)

# ------------------------------------------------ Abschnitt: BLP
st.markdown("## Das beste lineare Modell")
st.markdown(
    r"""
Die lineare Regression sagt $Y$ durch eine gewichtete Summe der Regressoren
vorher: $b'X = b_1 X_1 + \dots + b_p X_p$ (eine Konstante zählt als
Regressor mit). Gesucht sind die Gewichte mit dem kleinsten mittleren
quadratischen Vorhersagefehler. In der **Population**, also bei unbegrenzt
vielen Daten, heißt das Problem **Best Linear Prediction (BLP)**:

$$
\beta = \arg\min_{b \in \mathbb{R}^p} \; E\big(Y - b'X\big)^2 .
$$

Die Lösung zerlegt $Y$ in einen erklärten und einen unerklärten Teil,

$$
Y = \beta'X + \varepsilon, \qquad E[X\varepsilon] = 0,
$$

wobei die Bedingung $E[X\varepsilon]=0$ besagt, dass der Fehler
$\varepsilon$ mit keinem Regressor mehr korreliert ist: Linear ist aus $X$
nichts mehr herauszuholen. In der Praxis ersetzen wir die Erwartungswerte
durch Stichprobenmittel und erhalten die **Kleinste-Quadrate-Schätzung**
(OLS) $\hat\beta$, genau das, was `LinearRegression` in scikit-learn
ausrechnet.

Wie gut das Modell erklärt, misst der Anteil der erklärten Variation, das
**Bestimmtheitsmaß** $R^2 \in [0, 1]$: null heißt „das Modell erklärt
nichts“, eins heißt „das Modell erklärt alles“. Und hier lauert eine Falle,
die direkt an das Overfitting aus dem ersten Kapitel anschließt.
"""
)

# ------------------------------------------- Demo 1: Die R²-Falle
st.markdown("## Demo: Die R²-Falle")
st.markdown(
    r"""
Wir simulieren Daten mit einem echten, aber moderaten Zusammenhang: Die
**ersten fünf** Regressoren erklären gemeinsam genau 20 % der Variation
von $Y$ (das wahre $R^2$ ist also $0{,}2$), **alle weiteren** Regressoren
sind reines Rauschen ohne jeden Einfluss. Wir fitten eine lineare
Regression mit den ersten $p$ Regressoren auf $n = 100$ Beobachtungen und
messen das $R^2$ zweimal: auf den Trainingsdaten selbst (in-sample) und
auf frischen Testdaten (out-of-sample). Verschiebe $p$ und sieh zu, was
passiert.
"""
)

N_R2 = 100
WAHRES_R2 = 0.2

anz_regler, anz_knopf = st.columns([4, 1])
p_wahl = anz_regler.slider("Anzahl Regressoren p", 1, N_R2, 10)
with anz_knopf:
    st.markdown("&nbsp;")
    if st.button("Neue Daten", key="r2_neue_daten"):
        st.session_state["r2_seed"] = st.session_state.get("r2_seed", 2026) + 1
r2_seed = st.session_state.get("r2_seed", 2026)


@st.cache_data
def r2_kurven(n: int, seed: int):
    """Sample- und Test-R² über p; nur die ersten 5 Regressoren tragen
    Signal, kalibriert auf ein wahres R² von 0.2."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(n)
    # Var(Signal) = 5 * (0.5/√5)² = 0.25, Var(Rauschen) = 1 → R² = 0.2
    beta[:5] = 0.5 / np.sqrt(5)
    X = rng.standard_normal((n, n))
    y = X @ beta + rng.standard_normal(n)
    X_test = rng.standard_normal((2000, n))
    y_test = X_test @ beta + rng.standard_normal(2000)

    ps = list(range(1, n + 1))
    r2_in, r2_out = [], []
    for p in ps:
        modell = LinearRegression().fit(X[:, :p], y)
        r2_in.append(modell.score(X[:, :p], y))
        r2_out.append(modell.score(X_test[:, :p], y_test))
    return ps, r2_in, r2_out


ps, r2_in, r2_out = r2_kurven(N_R2, r2_seed)

metrik_in, metrik_out = st.columns(2)
metrik_in.metric("R² auf Trainingsdaten", f"{r2_in[p_wahl - 1]:.2f}")
metrik_out.metric("R² auf Testdaten", f"{r2_out[p_wahl - 1]:.2f}")

fig_r2 = go.Figure()
fig_r2.add_scatter(
    x=ps, y=r2_in, mode="lines", name="R² in-sample",
    line=dict(color=FARBEN["gletscher"], width=3),
)
fig_r2.add_scatter(
    x=ps, y=r2_out, mode="lines", name="R² out-of-sample",
    line=dict(color=FARBEN["sonne"], width=3),
)
fig_r2.add_hline(y=WAHRES_R2, line_dash="dash", line_color=FARBEN["schiefer"],
                 annotation_text="wahres R² = 0,2")
fig_r2.add_vline(x=p_wahl, line_dash="dot", line_color=FARBEN["schiefer"],
                 annotation_text="dein p", annotation_position="top")
fig_r2.update_layout(
    xaxis_title="Anzahl Regressoren p",
    yaxis_title="R²",
    yaxis=dict(range=[-1.05, 1.05]),
    height=420,
)
st.plotly_chart(fig_r2, use_container_width=True)

st.markdown(
    r"""
Bis etwa $p = 5$ lohnt jeder zusätzliche Regressor: Beide Kurven steigen,
denn das Modell sammelt echtes Signal ein. Danach trennen sich die Wege.
Das In-sample-$R^2$ klettert mechanisch immer weiter und erreicht bei
$p = n$ den Wert 1: Mit 100 Regressoren für 100 Beobachtungen lässt sich
**jede** Zielvariable perfekt „erklären“, obwohl 95 davon reines Rauschen
sind. Das Out-of-sample-$R^2$ schafft dagegen bestenfalls den wahren Wert
$0{,}2$, sinkt mit jedem überflüssigen Regressor und stürzt am Ende sogar
ins Negative, das überladene Modell sagt schlechter vorher als der bloße
Mittelwert. Die Lehre:

- Solange $p/n$ **klein** ist, ist das In-sample-$R^2$ ein brauchbares Maß,
  und Korrekturen wie das *adjustierte* $R^2$ (Skalierung mit
  $\tfrac{n}{n-p}$) reichen aus.
- Sobald $p/n$ **nicht klein** ist, hilft nur die ehrliche Messung auf
  Daten, die das Modell nie gesehen hat, genau das **Data Splitting** aus
  dem ersten Kapitel.

Was aber, wenn wir die vielen Regressoren gar nicht loswerden wollen, weil
in ihnen echte Information steckt? Dann braucht die Regression einen
eingebauten Schutz gegen Overfitting. Das ist das Thema des nächsten
Kapitels (**Lasso & Ridge**).
"""
)

merkkasten(
    "Merke",
    "Ein hohes R² auf den Trainingsdaten ist <b>kein</b> Qualitätsbeweis. "
    "Bei vielen Regressoren relativ zur Stichprobengröße lässt sich jede "
    "Zielvariable scheinbar erklären. Entscheidend ist die Vorhersagegüte "
    "auf ungesehenen Daten.",
    typ="merke",
)

# ------------------------------------- Abschnitt: Inference / FWL
st.markdown("## Von der Vorhersage zum Koeffizienten: Partialling-Out")
st.markdown(
    r"""
Nun zur zweiten Frage. Im Lohnbeispiel schreiben wir die Regression als

$$
Y = \beta_1 D + \beta_2' W + \varepsilon,
$$

mit $D$ als Indikator „Person ist eine Frau“ und $W$ als Kontrollen
(Ausbildung, Erfahrung usw.). Der Koeffizient $\beta_1$ beantwortet die
Inference Question: der Unterschied im **vorhergesagten** Lohn zwischen
Frauen und Männern **mit denselben Kontrollmerkmalen**. In der
Arbeitsökonomik heißt diese Größe **Gender Wage Gap**.

Was rechnet die Regression dabei eigentlich aus? Die Antwort gibt ein
klassisches Resultat, das **Frisch-Waugh-Lovell-Theorem (FWL)**. Man erhält
$\beta_1$ exakt, indem man in drei Schritten „herausrechnet“, was die
Kontrollen erklären (**Partialling-Out**):

1. Regressiere $Y$ auf $W$ und bilde das Residuum $\tilde{Y}$
   (der Lohn, bereinigt um alles, was $W$ linear erklärt).
2. Regressiere $D$ auf $W$ und bilde das Residuum $\tilde{D}$
   (das Zielmerkmal, bereinigt um $W$).
3. Regressiere $\tilde{Y}$ auf $\tilde{D}$. Die Steigung **ist** $\beta_1$.

„Kontrollieren für $W$“ ist also keine Metapher, sondern eine präzise
Rechenoperation: Der Koeffizient $\beta_1$ vergleicht nur noch die
Variation von $D$ und $Y$, die die Kontrollen *nicht* erklären können.
"""
)

# ------------------------------------- Demo 2: FWL / Gender Wage Gap
st.markdown("## Demo: Der Gender Wage Gap, dreimal gerechnet")
st.markdown(
    """
Wir simulieren einen Arbeitsmarkt, in dem der wahre Lohnnachteil von Frauen
bei gleichen Merkmalen **−2,00 €** pro Stunde beträgt. Zusätzlich
unterscheiden sich die Gruppen in ihren Merkmalen: Frauen haben im Schnitt
mehr Ausbildungsjahre, aber weniger Berufserfahrung (so ist es auch in den
US-Umfragedaten, an die dieses Beispiel angelehnt ist). Der Regler
steuert, wie stark sich die Gruppen unterscheiden. Drei Schätzer treten an:

1. **Naiv:** Differenz der mittleren Löhne, ohne jede Kontrolle
2. **Volle Regression:** Lohn auf Frau-Indikator *und* Kontrollen
3. **Partialling-Out (FWL):** die Drei-Schritt-Prozedur von oben
"""
)

regler_delta, knopf_fwl = st.columns([4, 1])
delta = regler_delta.slider(
    "Wie unterschiedlich sind die Gruppen in Ausbildung und Erfahrung?",
    0.0, 3.0, 1.5, step=0.5,
)
with knopf_fwl:
    st.markdown("&nbsp;")
    if st.button("Neue Stichprobe", key="fwl_neue_daten"):
        st.session_state["fwl_seed"] = st.session_state.get("fwl_seed", 0) + 1
fwl_seed = st.session_state.get("fwl_seed", 0)

WAHRER_GAP = -2.0


@st.cache_data
def wage_gap_schaetzer(delta: float, seed: int, n: int = 2500):
    """Simulierte Lohndaten; liefert naiven, vollen und FWL-Schätzer."""
    rng = np.random.default_rng(seed)
    frau = rng.binomial(1, 0.45, n)
    ausbildung = 12 + rng.normal(0, 2, n) + delta * frau
    erfahrung = np.clip(rng.normal(14, 6, n) - 1.5 * delta * frau, 0, None)
    lohn = (
        -6
        + 1.4 * ausbildung
        + 0.5 * erfahrung
        - 0.007 * erfahrung**2
        + WAHRER_GAP * frau
        + rng.normal(0, 3, n)
    )

    W = np.column_stack([ausbildung, erfahrung, erfahrung**2])

    naiv = lohn[frau == 1].mean() - lohn[frau == 0].mean()

    X_voll = np.column_stack([frau, W])
    voll = LinearRegression().fit(X_voll, lohn).coef_[0]

    lohn_tilde = lohn - LinearRegression().fit(W, lohn).predict(W)
    frau_tilde = frau - LinearRegression().fit(W, frau).predict(W)
    fwl = (
        LinearRegression()
        .fit(frau_tilde.reshape(-1, 1), lohn_tilde)
        .coef_[0]
    )
    return naiv, voll, fwl


naiv, voll, fwl = wage_gap_schaetzer(delta, fwl_seed)

spalte_naiv, spalte_voll, spalte_fwl = st.columns(3)
spalte_naiv.metric("1. Naiver Vergleich", f"{naiv:+.2f} €")
spalte_voll.metric("2. Volle Regression", f"{voll:+.2f} €")
spalte_fwl.metric("3. Partialling-Out", f"{fwl:+.2f} €")

fig_gap = go.Figure(
    go.Bar(
        x=["Naiver Vergleich", "Volle Regression", "Partialling-Out"],
        y=[naiv, voll, fwl],
        marker_color=[FARBEN["beere"], FARBEN["gletscher"], FARBEN["himmel"]],
        text=[f"{w:+.2f}" for w in (naiv, voll, fwl)],
        textposition="outside",
    )
)
fig_gap.add_hline(
    y=WAHRER_GAP, line_dash="dash", line_color=FARBEN["schiefer"],
    annotation_text="wahrer Unterschied −2,00 €",
)
fig_gap.update_layout(
    yaxis_title="Geschätzter Lohnunterschied (€/Stunde)",
    yaxis=dict(range=[-4.2, 1.5]),
    height=400,
    showlegend=False,
)
st.plotly_chart(fig_gap, use_container_width=True)

st.markdown(
    f"""
Zwei Beobachtungen. Erstens: Schätzer 2 und 3 sind **identisch** (hier
{voll:+.4f} vs. {fwl:+.4f}), genau das besagt das FWL-Theorem. Zweitens:
Der naive Vergleich verfehlt den wahren Wert, sobald sich die Gruppen in
lohnrelevanten Merkmalen unterscheiden, denn er vermengt den
Lohnunterschied *bei gleichen Merkmalen* mit den Merkmalsunterschieden
selbst. Stelle den Regler auf 0, und alle drei Schätzer treffen sich.

Übrigens: In den echten Daten (US Current Population Survey 2012, ledige
Beschäftigte) ergibt diese Rechnung einen Gender Wage Gap von etwa
**−2 $ pro Stunde** mit einem 95-%-Konfidenzintervall von rund −2,7 $ bis −1 $,
bei gleicher Erfahrung, Ausbildung und Region.
"""
)

merkkasten(
    "Achtung: kontrolliert heißt noch nicht kausal",
    "Der Koeffizient beschreibt den Unterschied der <b>Vorhersage</b> bei "
    "gleichen <b>erfassten</b> Merkmalen. Ob dahinter Diskriminierung, "
    "Selektion in unterschiedlich bezahlte Berufe oder ein nicht erfasster "
    "Störfaktor steckt, kann die Regression allein nicht entscheiden. Wann "
    "aus „kontrolliert“ tatsächlich „kausal“ wird, ist die Kernfrage der "
    "gesamten Kausalitäts-Sektion (Stichwort: DAGs und Confounding).",
    typ="achtung",
)

st.markdown(
    """
Merke dir das Partialling-Out gut: Im Kapitel **Kausales Machine Learning**
kehrt es als Herzstück von *Double Machine Learning* zurück, dort werden
die beiden Hilfsregressionen aus Schritt 1 und 2 durch flexible ML-Modelle
wie Random Forests ersetzt.
"""
)
st.page_link(
    "views/kausalitaet/kausales_ml.py",
    label="Vorgriff: Kausales Machine Learning (DML)",
    icon="🎯",
)
gruppen_aufgabe(
    "Worauf eure Gruppe hier aufbaut",
    [
        (
            "Partialling-Out nach Frisch-Waugh-Lovell ist der Kern von Double "
            "Machine Learning. Rechnet den Gender Wage Gap einmal klassisch "
            "und einmal mit einem ML-Modell für die Störfunktion und "
            "vergleicht die beiden Schätzungen."
        ),
        (
            "Standardfehler entscheiden über eure Aussage. Wann sind die "
            "klassischen Formeln falsch, und wie stark ändern robuste oder auf "
            "Gruppenebene geclusterte Standardfehler euer Konfidenzintervall?"
        ),
        (
            "Was passiert mit OLS, wenn es mehr Spalten als Zeilen gibt? Genau "
            "an dieser Stelle setzt das nächste Kapitel mit Lasso und Ridge an."
        ),
    ],
    hinweis=(
        "Startpunkt: <code>statsmodels</code> für Inferenz und robuste "
        "Standardfehler, <code>linearmodels</code> für Paneldaten und "
        "Clustering."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- G. James, D. Witten, T. Hastie & R. Tibshirani (2021), *An Introduction to Statistical Learning*, 2. Aufl., Springer, Kap. 3 (frei online)
- V. Chernozhukov, C. Hansen, N. Kallus, M. Spindler & V. Syrgkanis (2024), *Applied Causal Inference Powered by ML and AI*, Kap. 1 (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    """
Die R²-Falle hat gezeigt: Viele Regressoren überfordern die klassische
Regression. Das nächste Kapitel zeigt den Ausweg, mit dem sich hunderte
Regressoren zähmen lassen.
"""
)
weiter_lasso, weiter_kausal = st.columns(2)
with weiter_lasso:
    st.page_link(
        "views/ml/regularisierung.py", label="Weiter: Lasso & Ridge", icon="🎚️"
    )
with weiter_kausal:
    st.page_link(
        "views/kausalitaet/dags_confounding.py",
        label="Passt dazu: DAGs & Confounding",
        icon="🕸️",
    )
