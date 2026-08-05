"""Kapitel ML: Trees & Ensembles.

Decision Trees interaktiv, Random Forests und Feature Importance als
Grundlage für die Gruppenthemen Explainable ML und Causal ML.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.ml_demos import entscheidungsgrenze, monde_daten
from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten

kapitel_kopf(
    "🌲",
    "Trees & Ensembles",
    "Vorhersagen als Wenn-dann-Regeln und die Stärke der Aggregation",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Ein **Decision Tree** stellt nacheinander Fragen an die Daten: Liegt das
Einkommen über 3 000 €? Ist die Person jünger als 30? Jede Frage teilt die
Daten in zwei Gruppen, und am Ende jedes Astes steht eine Vorhersage. Je
nach Zielgröße spricht man von **Classification Trees** (kategoriales Label)
oder **Regression Trees** (stetiges Label).
"""
)

st.graphviz_chart(
    """
digraph {
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#EEF3FA",
          color="#3E6DB5", fontname="sans-serif"];
    edge [fontname="sans-serif", color="#5C6470", fontsize=11];
    A [label="Einkommen > 3 000 €?"];
    B [label="Schuldenquote < 40 %?"];
    C [label="Bürgschaft vorhanden?"];
    D [label="Kredit", fillcolor="#EDF6F0", color="#4C956C"];
    E [label="kein Kredit", fillcolor="#FCF1E9", color="#E8804C"];
    F [label="Kredit", fillcolor="#EDF6F0", color="#4C956C"];
    G [label="kein Kredit", fillcolor="#FCF1E9", color="#E8804C"];
    A -> B [label="ja"];
    A -> C [label="nein"];
    B -> D [label="ja"];
    B -> E [label="nein"];
    C -> F [label="ja"];
    C -> G [label="nein"];
}
""",
    use_container_width=True,
)

st.markdown(
    r"""
Was tut ein Tree mathematisch? Er zerlegt den Feature-Raum in $M$
rechteckige Regionen $R_1, \dots, R_M$ (die Blätter) und sagt in jeder
Region einen konstanten Wert vorher:

$$
\hat{g}(x) = \sum_{m=1}^{M} \hat{\beta}_m \, \mathbf{1}(x \in R_m),
$$

wobei $\hat{\beta}_m$ schlicht der **Mittelwert der Labels** in Region
$R_m$ ist (bei Klassifikation: die Mehrheitsklasse). Die Kunst liegt in der
Wahl der Regionen. Alle Zerlegungen durchzuprobieren ist aussichtslos,
deshalb arbeitet der Lernalgorithmus **rekursiv binär**: Er beginnt mit
allen Daten, sucht das eine Feature und den einen Schwellenwert, deren
Split das Fehlerkriterium am stärksten verbessert (bei Regression den MSE,
bei Klassifikation etwa die Gini-Impurity), und wiederholt das in beiden
Hälften, immer weiter, bis eine Stopp-Regel greift.

In echten Lohndaten sieht das zum Beispiel so aus: Der erste Split trennt
nach Hochschulabschluss, der zweite nach Berufserfahrung, und schon liest
man Vorhersagen wie „Abschluss und mehr als 9,5 Jahre Erfahrung → 24 $
Stundenlohn“ ab. Das macht Decision Trees außerordentlich
**interpretierbar**, denn jede Vorhersage lässt sich als Regelkette
vorlesen. Zugleich haben sie eine Schwäche, die dir aus dem Kapitel „Was
ist Maschinelles Lernen?“ vertraut vorkommen dürfte.
"""
)

# --------------------------------------- Demo 1: Baumtiefe & Overfitting
st.markdown("## Demo: Tiefe des Trees und Overfitting")
st.markdown(
    """
Die maximale **Tiefe** eines Decision Trees bestimmt seine Flexibilität,
denn jede zusätzliche Frageebene erlaubt feinere Aufteilungen des
Feature-Raums. In der Abbildung trennt ein Classification Tree zwei
ineinander verschränkte Punktwolken. Variiere die Tiefe und beobachte
sowohl die **Decision Boundary** als auch die Lücke zwischen Trainings- und
Testgenauigkeit.
"""
)

X, y = monde_daten(n=400, rauschen=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tiefe = st.slider("Maximale Tiefe (max_depth)", 1, 15, 2)

baum = DecisionTreeClassifier(max_depth=tiefe, random_state=0)
baum.fit(X_train, y_train)

fig_baum = entscheidungsgrenze(baum, X_train, y_train, titel=f"Decision Tree, Tiefe {tiefe}")
st.plotly_chart(fig_baum, use_container_width=True)

acc_train = baum.score(X_train, y_train)
acc_test = baum.score(X_test, y_test)
metrik_train, metrik_test = st.columns(2)
metrik_train.metric("Genauigkeit Training", f"{acc_train:.1%}")
metrik_test.metric("Genauigkeit Test", f"{acc_test:.1%}")

if tiefe <= 2:
    st.info(
        "**Underfitting:** Wenige achsenparallele Schnitte können die "
        "Mondform nicht abbilden. Erhöhe die Tiefe des Trees."
    )
elif tiefe >= 8:
    st.warning(
        "**Overfitting:** Der Tree bildet kleinste Regionen um einzelne "
        "Rauschpunkte. Die Trainingsgenauigkeit nähert sich 100 %, während "
        "die Testgenauigkeit zurückfällt. Es ist dieselbe Systematik wie bei "
        "der Polynomregression."
    )

st.markdown(
    """
Wie tief soll der Tree also werden? Man kann die Tiefe vorab festlegen
oder Stopp-Regeln setzen (etwa eine Mindestzahl von Beobachtungen pro
Blatt). Eleganter ist **Pruning**: erst einen großen Tree wachsen lassen,
dann die Äste zurückschneiden, deren Beitrag den Komplexitätszuwachs nicht
rechtfertigt, wobei die Schnitttiefe per Cross-Validation gewählt wird
(*Cost-Complexity Pruning*). In der Praxis bleibt aber auch ein sauber
gestutzter Einzelbaum meist ein mittelmäßiger Prädiktor: Seine
stückweise konstante Vorhersage ist eine grobe Näherung, und kleine
Änderungen in den Daten können die Splitfolge und damit den gesamten Tree
umwerfen. Genau diese Instabilität machen sich die Ensembles zunutze.
"""
)

# ---------------------------------------------- Demo 2: Random Forest
st.markdown("## Demo: Vom Tree zum Random Forest")
st.markdown(
    r"""
Die zentrale Idee der **Ensembles**: viele *unterschiedliche* Trees
trainieren und ihre Vorhersagen mitteln beziehungsweise per
Mehrheitsentscheid aggregieren. Doch woher nimmt man viele
unterschiedliche Trees, wenn es nur einen Datensatz gibt? Die Antwort ist
der **Bootstrap**: Man zieht aus den $n$ Beobachtungen $n$-mal *mit
Zurücklegen* und erhält so eine Quasi-Kopie des Datensatzes, in der manche
Beobachtungen mehrfach vorkommen und andere fehlen. Auf jeder solchen
Bootstrap-Stichprobe wächst ein eigener tiefer Tree; das Mitteln über
alle nennt man **Bagging** (*Bootstrap Aggregation*). Der **Random
Forest** fügt eine zweite Zufallsquelle hinzu: An jedem Split steht nur
eine zufällige Teilmenge der Features zur Auswahl, damit sich die Trees
auch strukturell unterscheiden.

Warum die Aggregation hilft, zeigt eine kurze Rechnung. Für $B$ Schätzer mit
Varianz $\sigma^2$ und paarweiser Korrelation $\rho$ gilt

$$
\mathrm{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)\right)
= \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

Der zweite Term verschwindet mit wachsender Ensemblegröße $B$, übrig bleibt
allein der korrelierte Anteil $\rho\,\sigma^2$. Genau deshalb dekorreliert
der Random Forest seine Trees durch doppelte Zufälligkeit in Daten *und*
Features: Je kleiner $\rho$, desto stärker die Varianzreduktion, und das bei
unverändertem Bias.
"""
)

anzahl_baeume = st.select_slider(
    "Anzahl Trees (n_estimators)", options=[1, 2, 5, 10, 25, 50, 100, 200], value=1
)


@st.cache_data
def wald_trainieren(n_baeume: int, X_tr, y_tr):
    wald = RandomForestClassifier(n_estimators=n_baeume, random_state=0)
    wald.fit(X_tr, y_tr)
    return wald


@st.cache_data
def wald_lernkurve(X_tr, y_tr, X_te, y_te):
    stufen = [1, 2, 5, 10, 25, 50, 100, 200]
    return stufen, [
        RandomForestClassifier(n_estimators=s, random_state=0)
        .fit(X_tr, y_tr)
        .score(X_te, y_te)
        for s in stufen
    ]


wald = wald_trainieren(anzahl_baeume, X_train, y_train)

spalte_grenze, spalte_kurve = st.columns(2)
with spalte_grenze:
    st.plotly_chart(
        entscheidungsgrenze(
            wald, X_train, y_train, titel=f"Random Forest, {anzahl_baeume} Trees"
        ),
        use_container_width=True,
    )
with spalte_kurve:
    stufen, genauigkeiten = wald_lernkurve(X_train, y_train, X_test, y_test)
    fig_kurve = go.Figure()
    fig_kurve.add_scatter(
        x=stufen, y=genauigkeiten, mode="lines+markers", name="Testgenauigkeit",
        line=dict(color=FARBEN["wiese"], width=3),
    )
    fig_kurve.add_vline(x=anzahl_baeume, line_dash="dot", line_color=FARBEN["schiefer"])
    fig_kurve.update_layout(
        title="Testgenauigkeit über die Anzahl Trees",
        xaxis_title="Anzahl Trees (log)", xaxis_type="log",
        yaxis_title="Genauigkeit", height=420,
    )
    st.plotly_chart(fig_kurve, use_container_width=True)

st.markdown(
    """
Mit einem einzelnen Tree ist die Decision Boundary zerklüftet und instabil.
Mit vielen Trees wird sie glatt, und die Testgenauigkeit steigt, obwohl
jeder einzelne Tree tief und damit anfällig für Overfitting ist. Beachte
die Arbeitsteilung: Die *Tiefe* der Einzelbäume hält den Bias klein, das
*Mitteln* drückt die Varianz. Der Random Forest kuriert also genau die
Schwäche, an der der Einzelbaum krankt.
"""
)

# ------------------------------------------------ Demo 3: Boosting
st.markdown("## Gradient Boosting: Fehler nacheinander korrigieren")
st.markdown(
    r"""
Die zweite große Ensemble-Strategie geht den umgekehrten Weg. **Boosting**
mittelt nicht viele tiefe Trees parallel, sondern baut **flache** Trees
*nacheinander*, wobei jeder neue Tree die Fehler des bisherigen Ensembles
korrigiert:

1. Starte mit der Vorhersage null; die Residuen sind zunächst die Labels
   selbst: $R_i = Y_i$.
2. Für $j = 1, \dots, J$: Fitte einen flachen Tree $\hat{g}_j$ auf die
   aktuellen Residuen und aktualisiere
   $R_i \leftarrow R_i - \eta\, \hat{g}_j(x_i)$.
3. Die Gesamtvorhersage ist die Summe aller Korrekturen:
   $\hat{g}(x) = \sum_{j=1}^{J} \eta\, \hat{g}_j(x)$.

Die **Lernrate** $\eta \in (0, 1]$ dosiert, wie viel jeder Tree
beitragen darf; die Anzahl Schritte $J$ und $\eta$ wählt man per
Cross-Validation. Flache Trees halten hier die Varianz klein, und jeder
Schritt knabbert am verbleibenden Bias. Probiere es aus:
"""
)

regler_schritte, regler_lernrate = st.columns(2)
schritte = regler_schritte.select_slider(
    "Anzahl Boosting-Schritte J", options=[1, 2, 3, 5, 10, 25, 50, 100, 200], value=2
)
lernrate = regler_lernrate.slider("Lernrate η", 0.1, 1.0, 0.3, step=0.1)


@st.cache_data
def boosting_sequenz(lernrate: float, max_schritte: int = 200, seed: int = 2026):
    """Boosting mit flachen Regression Trees auf 1D-Sinusdaten.

    Liefert die kumulierte Vorhersage auf einem Raster nach jedem Schritt
    sowie Trainings- und Testfehler je Schritt.
    """
    rng = np.random.default_rng(seed)
    x_tr = np.sort(rng.uniform(0, 1, 120))
    y_tr = np.sin(2 * np.pi * x_tr) + rng.normal(0, 0.3, 120)
    x_te = np.sort(rng.uniform(0, 1, 200))
    y_te = np.sin(2 * np.pi * x_te) + rng.normal(0, 0.3, 200)
    raster = np.linspace(0, 1, 300)

    fit_tr = np.zeros_like(x_tr)
    fit_te = np.zeros_like(x_te)
    fit_raster = np.zeros_like(raster)
    kurven, mse_tr, mse_te = [], [], []
    for _ in range(max_schritte):
        baum = DecisionTreeRegressor(max_depth=2, random_state=0)
        baum.fit(x_tr.reshape(-1, 1), y_tr - fit_tr)
        fit_tr += lernrate * baum.predict(x_tr.reshape(-1, 1))
        fit_te += lernrate * baum.predict(x_te.reshape(-1, 1))
        fit_raster += lernrate * baum.predict(raster.reshape(-1, 1))
        kurven.append(fit_raster.copy())
        mse_tr.append(float(np.mean((y_tr - fit_tr) ** 2)))
        mse_te.append(float(np.mean((y_te - fit_te) ** 2)))
    return x_tr, y_tr, raster, kurven, mse_tr, mse_te


x_boost, y_boost, raster_boost, kurven, mse_tr, mse_te = boosting_sequenz(lernrate)

spalte_fit, spalte_fehler = st.columns(2)
with spalte_fit:
    fig_boost = go.Figure()
    fig_boost.add_scatter(
        x=x_boost, y=y_boost, mode="markers", name="Trainingsdaten",
        marker=dict(color=FARBEN["gletscher"], size=7, opacity=0.7),
    )
    fig_boost.add_scatter(
        x=raster_boost, y=np.sin(2 * np.pi * raster_boost), mode="lines",
        name="Wahrer Zusammenhang",
        line=dict(color=FARBEN["schiefer"], dash="dash", width=2),
    )
    fig_boost.add_scatter(
        x=raster_boost, y=kurven[schritte - 1], mode="lines",
        name=f"Ensemble nach {schritte} Schritten",
        line=dict(color=FARBEN["beere"], width=3),
    )
    fig_boost.update_layout(
        title="Kumulierte Boosting-Vorhersage",
        xaxis_title="Feature x", yaxis_title="Label y",
        yaxis=dict(range=[-2.2, 2.2]), height=420,
    )
    st.plotly_chart(fig_boost, use_container_width=True)
with spalte_fehler:
    schritt_achse = list(range(1, len(mse_tr) + 1))
    fig_boost_mse = go.Figure()
    fig_boost_mse.add_scatter(
        x=schritt_achse, y=mse_tr, mode="lines", name="Trainingsfehler",
        line=dict(color=FARBEN["gletscher"], width=3),
    )
    fig_boost_mse.add_scatter(
        x=schritt_achse, y=mse_te, mode="lines", name="Testfehler",
        line=dict(color=FARBEN["sonne"], width=3),
    )
    fig_boost_mse.add_vline(
        x=schritte, line_dash="dot", line_color=FARBEN["schiefer"]
    )
    fig_boost_mse.update_layout(
        title="Fehler pro Boosting-Schritt",
        xaxis_title="Schritte J (log)", xaxis_type="log",
        yaxis_title="MSE", height=420,
    )
    st.plotly_chart(fig_boost_mse, use_container_width=True)

st.markdown(
    """
Nach einem Schritt ist die Vorhersage eine grobe Treppe, nach wenigen
Schritten schmiegt sie sich an die Sinuskurve, und bei sehr vielen
Schritten mit hoher Lernrate beginnt das Ensemble, dem Rauschen
hinterherzulaufen: Der Testfehler dreht wieder nach oben, während der
Trainingsfehler weiter sinkt. Auch Boosting ist also nicht immun gegen
Overfitting, es overfittet nur *langsam und kontrolliert*. Die bekanntesten
Implementierungen dieser Idee, **XGBoost** und **LightGBM**, gelten auf
tabellarischen Daten häufig als Stand der Technik.
"""
)

# ------------------------------------------ Demo 4: Feature Importance
st.markdown("## Demo: Feature Importance")
st.markdown(
    """
Ein Random Forest aus 200 Trees ist keine vorlesbare Regelkette mehr. Er
kann jedoch ausweisen, welche Features seine Vorhersagen am stärksten
beeinflussen (**Feature Importance**): Wie oft und mit welchem Gewinn an
Trennschärfe wurde über ein Feature gesplittet? Die folgende Abbildung
zeigt dies für ein fiktives Kreditausfall-Modell.
"""
)


@st.cache_data
def kredit_importances(seed: int = 5):
    namen = ["Einkommen", "Schuldenquote", "Alter", "Anzahl Kredite", "Kontostand"]
    X_k, y_k = make_classification(
        n_samples=800, n_features=5, n_informative=3, n_redundant=1,
        random_state=seed, shuffle=False,
    )
    wald_k = RandomForestClassifier(n_estimators=200, random_state=0)
    wald_k.fit(X_k, y_k)
    return namen, wald_k.feature_importances_


namen, wichtigkeiten = kredit_importances()
reihenfolge = np.argsort(wichtigkeiten)
fig_imp = go.Figure(
    go.Bar(
        x=wichtigkeiten[reihenfolge],
        y=[namen[i] for i in reihenfolge],
        orientation="h",
        marker_color=FARBEN["gletscher"],
    )
)
fig_imp.update_layout(
    xaxis_title="Feature Importance", height=320, margin=dict(t=20)
)
st.plotly_chart(fig_imp, use_container_width=True)

merkkasten(
    "Achtung: wichtig ist nicht ursächlich",
    "Feature Importance besagt, dass das Modell ein Feature stark für seine "
    "Vorhersagen nutzt. Sie besagt <b>nicht</b>, dass dieses Feature das "
    "Ergebnis verursacht. Ein Confounder oder ein bloßes Korrelat kann an "
    "erster Stelle stehen. Diese Unterscheidung wird im Kapitel Explainable "
    "ML und in der gesamten Kausalitäts-Sektion zentral sein.",
    typ="achtung",
)
gruppen_aufgabe(
    "Worauf eure Gruppe hier aufbaut",
    [
        (
            "Feature Importance ist keine Kausalität. Konstruiert euch ein "
            "Beispiel, in dem ein Merkmal hohe Importance bekommt, obwohl es "
            "nichts verursacht, etwa weil es mit der wahren Ursache korreliert."
        ),
        (
            "Ein Random Forest lässt sich zu einem <b>Causal Forest</b> "
            "umbauen. Was genau muss man dafür ändern? Stichworte sind Honest "
            "Splitting und ein anderes Splitting-Kriterium."
        ),
        (
            "Boosting hat viele Hyperparameter. Wie tuned man sie, ohne sich "
            "das Testset zu verbrennen, und wie viel Genauigkeit kostet ein "
            "ehrliches Vorgehen tatsächlich?"
        ),
    ],
    hinweis=(
        "Startpunkt: <code>scikit-learn</code>, dazu <code>xgboost</code> "
        "oder <code>lightgbm</code>. Für den Schritt zu Causal Forests "
        "<code>econml</code> und Athey &amp; Imbens (2016)."
    ),
)


# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- G. James, D. Witten, T. Hastie & R. Tibshirani (2021), *An Introduction to Statistical Learning*, 2. Aufl., Springer, Kap. 8 (frei online)
- T. Hastie, R. Tibshirani & J. Friedman (2009), *The Elements of Statistical Learning*, 2. Aufl., Springer, Kap. 9, 10 und 15 (frei online)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_xai, weiter_nn = st.columns(2)
with weiter_xai:
    st.page_link(
        "views/ml/explainable_ml.py",
        label="Weiter: Explainable ML",
        icon="🔍",
    )
with weiter_nn:
    st.page_link(
        "views/ml/neuronale_netze.py", label="Oder: Neural Networks", icon="🧠"
    )
