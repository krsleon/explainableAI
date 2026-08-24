""" Explainable AI App
Frage: Wie koennen wir die Art von Pinguinen erkennen 
und die Entscheidung des KI-Modells nachvollziehen?"""

import sys
from pathlib import Path
import os

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Der Projektordner liegt nicht automatisch im Suchpfad, weil Streamlit die
# Seite aus dem Hauptverzeichnis heraus startet. Diese drei Zeilen braucht
# ihr, sobald ihr eigenen Code aus Nachbardateien importieren wollt.
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

import analyse  # noqa: E402  (erst nach dem sys.path-Eintrag importierbar)
from utils.theming import FARBEN, merkkasten

data = analyse.load_data()
model, X_test, y_test, y_pred = analyse.train_blackbox(data, report=False)

st.markdown("# Explainable AI: Wie können wir Black-Box Modelle verstehen?")
st.caption(
    "Projekt von Katharina Gudat und Leon Kraus · Daten: Palmer Penguins, 2014, 344 Penguins"
)

merkkasten(
    "Dieses Projekt veranschaulicht zwei Modelle, die Black-Box Entscheidungen nachvollziehbar machen.",
    "Vor allem geht es um <b>Lime</b> und <b>SHAP</b>. "
    "Tab 1 ist eine Einleitung in Explainable AI, Tab 2 zeigt das Pinguin-Datenset, Tab 3 beschreibt das verwendete Blackbox-Modell, Tab 4 behandelt die LIME-Methode, Tab 5 die SHAP-Methode, und Tab 6 geht kurz auf andere Möglichkeiten der Explainable AI ein.",
    typ="merke",
)

tab_intro, tab_daten, tab_blackbox, tab_lime, tab_shap, tab_ausblick = st.tabs(
    [
        "1 · Einleitung",
        "2 · Daten: Palmer Penguins",
        "3 · Das Blackbox-Modell",
        "4 · LIME",
        "5 · SHAP",
        "6 · Ausblick: Andere Möglichkeiten",
    ]
)
#============================================================= Intro to xAI
with tab_intro:
    st.markdown("## 1 · Einleitung: Explainable AI")
    st.markdown(
        "Explainable AI (xAI) ist ein Forschungsgebiet, das sich mit der Nachvollziehbarkeit von Entscheidungen von KI-Modellen beschäftigt. "
        "Es geht darum, die Funktionsweise von Modellen zu verstehen und ihre Entscheidungen zu erklären."
        "Dazu gibt es verschiedene Herangehensweisen, die schematisch in der folgenden Abbildung dargestellt werden."
    )
    st.image(os.path.join(ORDNER, "images", "xAI_overview.png"), caption="Abbildung 1: Überblick über Explainable AI Methoden")
    st.markdown(" Die erste Unterscheidung in der Grafik meint den Unterschied zwischen intrinsisch interpretierbaren Modellen (z. B. lineare Regression, Entscheidungsbäume) und post-hoc Methoden"
                ", die im Nachhinein auf kompliziertere Modelle (z. B. Random Forest, Neural Networks) angewendet werden können."
            "Klassische, lineare Machine Learning Modelle bieten eine gewisse Interpretierbarkeit in ihren Entscheidungen, indem Koeffizienten und Verzweigungen direkt abzulesen sind, sie sind also erklärbar by design."
        "Bei hochdimensionalen Modellen hingegen leidet die Interpretierbarkeit unter der besseren Performance – die Millionen nicht-linear verknüpften Parameter entziehen sich der menschlichen Intuition."
    )
    st.markdown(
        "Aus diesem Grund spricht man von *Blackbox-Modellen*. "
    )
    st.markdown(
        """
        ## Darum suchen wir Ansätze, um Ergebnisse interpretierbar zu machen...

        Ein reines Vorhersageergebnis reicht in der Praxis selten aus:
        - **Vertrauen & Validierung:** Trifft das Modell Entscheidungen anhand biologisch plausibler Merkmale oder verlässt es sich auf zufällige Artefakte im Datensatz?
        - **Fehlersuche & Debugging:** Warum wurde ein bestimmter Pinguin falsch klassifiziert? Welches Merkmal hat den Ausschlag gegeben?
        - **Verantwortung & Nachvollziehbarkeit:** In kritischen Anwendungen müssen Entscheidungen gegenüber Anwendern und Regulatoren begründet werden können.

        Um den Konflikt zwischen hoher Vorhersageleistung und mangelnder Transparenz aufzulösen, setzt **Explainable AI (XAI)** unter anderem auf *modellagnostische Post-Hoc-Erklärungen*, also Verfahren, die auf beliebige Modelle angewandt werden können. 
        Dabei wird das Verhalten der Black Box von außen beobachtet und analysiert – wie in den folgenden Tabs an zwei führenden Verfahren demonstriert wird:
        """
    )

    merkkasten(
        "Ansätze für mehr Erklärbarkeit",
        "• <b>LIME (Tab 4):</b> Erklärt einzelne Vorhersagen lokal durch ein einfaches lineares Ersatzmodell in der direkten Nachbarschaft eines Datenpunkts.<br>"
        "• <b>SHAP (Tab 5):</b> Nutzt Konzepte der kooperativen Spieltheorie, um den exakten, fairen Beitrag jedes einzelnen Merkmals zur Entscheidung zu quantifizieren.",
        typ="definition",
    )
    
    st.markdown("LIME und SHAP sind lokale Methoden, die Auskunft über die Entscheidung eines Modells in der Nähe eines bestimmten Datenpunkts geben. Die modell-agnostischen Methoden können auch global sein und Auskunft über das Gesamtverhalten des Modells geben. ")
    st.markdown("Es existieren auch modell-spezifische Ansätze, die auf bestimmte Modelltypen zugeschnitten sind, z. B. neuronale Netze oder Entscheidungsbäume. Diese werden in Tab 6 kurz vorgestellt.")


#============================================================= Daten
with tab_daten:
    st.markdown("## 2 · Daten: Palmer Penguins")
    st.caption(
        "Die Daten stammen aus dem Datensatz 'Palmer Penguins' von Allison Horst, 2014. "
        "Er enthält Informationen zu 344 Pinguinen von drei Arten (Adelie, Chinstrap, Gentoo) "
        "und vier Inseln (Biscoe, Dream, Torgersen, und Palmer)."
    )
    st.markdown(
        "Die Daten enthalten die folgenden Spalten: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`."
    )
    st.dataframe(data.sample(7, random_state=42))
    st.markdown(
        "Wir werden die Spalten `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, und `body_mass_g` als Features verwenden, um die Art der Pinguine vorherzusagen. "
        "Im Plot unten können wir die Verteilung der Pinguinarten in Abhängigkeit von zwei ausgewählten Features visualisieren."
        "Es fällt auf, dass die Arten je nach ausgewählten Features unterschiedlich gut trennbar sind. "
        "Unser Blackbox-Modell soll die Art der Pinguine anhand der Features vorhersagen, und wir wollen danach nachvollziehen, warum eine bestimmte Klassifikation stattgefunden hat."
    )
    numeric_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
    ]

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "x-axis",
            numeric_features,
            index=0
        )

    with col2:
        y_feature = st.selectbox(
            "y-axis",
            numeric_features,
            index=3
        )

    fig = px.scatter(
        data,
        x=x_feature,
        y=y_feature,
        color="species",
        hover_data=["sex", "island"],
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
#============================================================= Blackbox-Modell
with tab_blackbox:
    st.markdown("## 3 · Das Blackbox-Modell")
    st.markdown("Wir trainieren ein Random Forest Modell, um die Art der Pinguine anhand der vier ausgewählten Features vorherzusagen. "
                "Dafur verwenden wir die sklearn Bibliothek. Das Modell wird auf 70% der Daten trainiert und auf den restlichen 30% getestet. ")
    accuracy = accuracy_score(y_test, y_pred)

    st.metric(
        "Accuracy",
        f"{accuracy:.1%}"
    )
    cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
    )

    fig_cm = px.imshow(
        cm,
        x=model.classes_,
        y=model.classes_,
        text_auto=True,
        labels={
            "x": "Vorhergesagte Spezies",
            "y": "Tatsächliche Spezies",
            "color": "Anzahl"
        },
        title="Confusion Matrix"
    )

    fig_cm.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )
    importance_df = pd.DataFrame({
    "Feature": analyse.standard_features,
    "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )
    fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="Feature Importance des Random Forests",
    labels={
        "Importance": "Importance",
        "Feature": "Feature"
    }
    )
    
    fig_importance.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )
    
    fig_importance.update_layout(
        xaxis_range=[0, importance_df["Importance"].max() * 1.15],
        height=400
    )
    
    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )
    


#============================================================= LIME
with tab_lime:
    st.markdown("## 4 · LIME: Local Interpretable Model-agnostic Explanations")
    st.caption(
        "LIME ist eine Methode, die lokale Erklärungen für die Vorhersagen von Black-Box-Modellen liefert. "
    )
    st.markdown(
        "Die Grundidee von LIME ist, dass wir ein komplexes Modell durch ein einfaches, interpretiertes Modell approximieren können, "
        "das in der Nähe der Vorhersage des komplexen Modells gut funktioniert. "
        "Dazu werden zufällige Perturbationen der Eingabedaten erzeugt und die Vorhersagen des Black-Box-Modells für diese Perturbationen gesammelt. "
        "Anschließend wird ein einfaches Modell (z.B. lineares Modell, Decision Tree) auf diesen Daten trainiert, um die Vorhersage des Black-Box-Modells zu erklären."
        "Es handelt sich um eine lokale Methode, da eine Erklärung nur für eine einzelne Vorhersage/Instanz - in unserem Beispiel für einen Pinguin - erzeugt wird, und nicht für das gesamte Modell."
    )
    
    st.markdown("### Mathematische Grundlagen von LIME")

    st.markdown(
        r"""
        Für eine einzelne Instanz $x$ versucht LIME, das Verhalten des komplexen
        Black-Box-Modells $f$ in einer lokalen Umgebung durch ein einfaches,
        interpretierbares Modell $g$ zu approximieren. Für unsere Anwendung ist
        $f$ der Random Forest und $x$ ein einzelner Pinguin.

        Als einfaches Erklärungsmodell kann beispielsweise ein lineares Modell
        verwendet werden:
        """
    )

    st.latex(
        r"""
        g(x') = \beta_0 + \sum_{j=1}^{p} \beta_j x'_j .
        """
    )

    st.markdown(
        r"""
        Dabei beschreibt $x'$ die Merkmale einer (möglicherweise perturbierten)
        Instanz und $\beta_j$ gibt an, welchen lokalen Einfluss das Merkmal
        $j$ auf die Vorhersage des Surrogatmodells hat. Ein großer Betrag
        $|\beta_j|$ bedeutet somit einen starken lokalen Einfluss. Das Vorzeichen
        gibt die Richtung des Einflusses an: Ein positiver Koeffizient unterstützt
        die betrachtete Klasse, ein negativer Koeffizient spricht gegen sie.

        Wichtig ist dabei, dass diese Koeffizienten <b>keine globalen
        Feature-Importances des Random Forests</b> sind. Sie beschreiben nur,
        wie das vereinfachte Modell das Verhalten des Random Forests in der
        Umgebung der ausgewählten Instanz approximiert.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        r"""
        Um die lokale Umgebung zu erzeugen, verändert LIME die Merkmale der
        ursprünglichen Instanz und lässt das Black-Box-Modell die erzeugten
        Datenpunkte klassifizieren. Datenpunkte, die näher an der ursprünglichen
        Instanz liegen, sollen dabei stärker zur lokalen Erklärung beitragen.
        """
    )
    st.markdown("#### Optimierungsproblem von LIME")
    st.markdown(
    r"""
    LIME sucht dabei ein möglichst einfaches Modell $g$, das das Verhalten
    des Black-Box-Modells $f$ in der lokalen Umgebung von $x$ gut beschreibt:

    """
    )

    st.latex(
        r"""
        \xi(x)
        =
        \underset{g\in G}{\operatorname{arg\,min}}
        \left[
            L(f,g,\pi_x) + \Omega(g)
        \right].
        """
    )

    st.markdown(
        r"""
        Dabei misst $L(f,g,\pi_x)$, wie gut das Surrogatmodell $g$ das
        Black-Box-Modell in der lokalen Umgebung reproduziert, während
        $\Omega(g)$ die Komplexität des Erklärungsmodells bestraft.
        LIME sucht also einen Kompromiss zwischen **Genauigkeit der lokalen
        Approximation** und **Einfachheit der Erklärung**.
        """
    )

    st.markdown("#### Kernel und Kernel-Breite")

    st.markdown(
        r"""
        Die Nähe eines perturbierten Datenpunkts $x'$ zur ursprünglichen Instanz
        $x$ wird durch einen Kernel gewichtet. Vereinfacht kann man sich die
        Gewichtsfunktion als
        """
    )

    st.latex(
        r"""
        \pi_x(x') =
        \exp\left(
        -\frac{D(x,x')^2}{\sigma^2}
        \right)
        """
    )

    st.markdown(
        r"""
        vorstellen. Hier bezeichnet $D(x,x')$ einen Distanzmaß zwischen den
        beiden Instanzen und $\sigma$ die <b>Kernel-Breite</b>
        (<i>kernel width</i>).

        Die Kernel-Breite bestimmt damit, wie groß die lokale Umgebung ist:
        """
    )

    st.markdown(
        r"""
        - **Kleine Kernel-Breite:** Nur Datenpunkte, die sehr nahe an der
          ursprünglichen Instanz liegen, erhalten ein hohes Gewicht.
          Die Erklärung ist dadurch stärker lokalisiert.
        - **Große Kernel-Breite:** Auch weiter entfernte Datenpunkte erhalten
          noch ein relevantes Gewicht. Die Erklärung berücksichtigt dadurch
          einen größeren Bereich des Datenraums.

        Die Wahl der Kernel-Breite beeinflusst daher direkt die resultierende
        Erklärung. Es gibt nicht notwendigerweise eine einzige, eindeutig
        "richtige" lokale Erklärung.
        """,
        unsafe_allow_html=True
    )

    if "lime_instance_idx" not in st.session_state:
        st.session_state["lime_instance_idx"] = 7

    # Bounds check
    if st.session_state["lime_instance_idx"] < 0:
        st.session_state["lime_instance_idx"] = 0
    elif st.session_state["lime_instance_idx"] >= len(X_test):
        st.session_state["lime_instance_idx"] = len(X_test) - 1

    def _prev_penguin():
        if st.session_state["lime_instance_idx"] > 0:
            st.session_state["lime_instance_idx"] -= 1

    def _next_penguin():
        if st.session_state["lime_instance_idx"] < len(X_test) - 1:
            st.session_state["lime_instance_idx"] += 1

    st.markdown("### Ihr seid dran: LIME-Vorhersage für einen Pinguin")
    col_prev, col_num, col_next = st.columns([1, 2, 1], vertical_alignment="center")

    with col_prev:
        st.button(
            "◀ Vorheriger",
            key="lime_btn_prev",
            on_click=_prev_penguin,
            disabled=(st.session_state["lime_instance_idx"] <= 0),
            use_container_width=True,
        )

    with col_num:
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<span style='font-size: 1.8rem; font-weight: 700;'>Pinguin #{st.session_state['lime_instance_idx']}</span>"
            f"<br><span style='font-size: 0.85rem; color: #888;'>Index {st.session_state['lime_instance_idx']} von {len(X_test) - 1}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        st.button(
            "Nächster ▶",
            key="lime_btn_next",
            on_click=_next_penguin,
            disabled=(st.session_state["lime_instance_idx"] >= len(X_test) - 1),
            use_container_width=True,
        )

    instance_idx = st.session_state["lime_instance_idx"]
    instance_to_explain = X_test.iloc[instance_idx].to_numpy()

    actual_class = y_test.iloc[instance_idx]

    predicted_class = model.predict(
        instance_to_explain.reshape(1, -1)
    )[0]

    st.write(f"**Tatsächliche Art:** {actual_class}")
    st.write(f"**Vorhergesagte Art:** {predicted_class}")
    st.markdown("Eigenschaften des gewählten Pinguins:")
    st.dataframe(
    X_test.iloc[[instance_idx]],
    hide_index=True,
    use_container_width=True
    )
    st.markdown("### Kernel-Abhängigkeit von LIME Erklärungen")
    kernel_width = st.slider(
        "Kernel width - Wie lokal ist unsere Erklärung?",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    explainer = analyse.create_lime_explainer(model, data, kernel_width)
    explanation = analyse.explain_instance(explainer, model, instance_to_explain)
    lime_values = explanation.as_list(explanation.top_labels[0])
    
    # Convert LIME output into a DataFrame
    lime_df = pd.DataFrame(
        lime_values,
        columns=["Feature", "Beitrag zur Vorhersage"]
    )

    # Sort so the strongest contributions appear at the top
    lime_df = lime_df.sort_values(
        "Beitrag zur Vorhersage",
        ascending=True
    )

    fig = px.bar(
        lime_df,
        x="Beitrag zur Vorhersage",
        y="Feature",
        orientation="h",
        title=f"LIME Vorhersage — {predicted_class}",
        labels={
            "Contribution": "Beitrag zur Vorhersage",
            "Feature": "Feature"
        }
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_width=2
    )

    fig.update_layout(
        title=(
            f"LIME Erklärung für {predicted_class}"
            f"<br><sup>Kernelbreite = {kernel_width}</sup>"
        ),
        height=400,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
    
    st.markdown("Im Beispiel könnt ihr sehen, dass die Kernelbreite die Bedeutung der einzelnen Features für die Vorhersage verändert."
                "Bei zu geringer Kernelbreite kann überhaupt keine signifikante Kontribution festgestellt werden (Werte von $10^{-21}$)."
                "Teilweise ändern sich die aufgeführten wichtigsten Features bei steigender Kernelbreite.")
    
    st.markdown("### Stärken und Schwächen von LIME")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
            **✓ Stärken**

            - **Modellunabhängig:** LIME benötigt keinen direkten Zugriff auf
              die interne Struktur des Black-Box-Modells.

            - **Lokale Erklärungen:** Es erklärt konkrete einzelne
              Vorhersagen und kann dadurch sehr detaillierte Einzelfälle
              untersuchen.

            - **Intuitiv:** Die Beiträge einzelner Merkmale können als
              positive oder negative Gewichte dargestellt werden.

            - **Vielseitig:** Das Grundprinzip kann auf unterschiedliche
              Datentypen und Modelle angewendet werden.

            - **Einfach visualisierbar:** Die lokalen Feature-Beiträge lassen
              sich beispielsweise als Balkendiagramm darstellen.
            """
        )

    with col2:
        st.warning(
            """
            **⚠ Schwächen**

            - **Lokale Methode:** Eine Erklärung für einen Pinguin sagt
              nicht automatisch etwas über andere Pinguine aus.

            - **Abhängigkeit von Hyperparametern:** Die Erklärung kann sich
              beispielsweise mit der Kernel-Breite oder der Anzahl der
              erzeugten Perturbationen verändern.

            - **Stochastisch:** Durch die zufällige Erzeugung von
              Perturbationen können sich Erklärungen zwischen verschiedenen
              Läufen unterscheiden.

            - **Surrogatmodell ist nur eine Approximation:** Das einfache
              Modell muss das Black-Box-Modell nicht außerhalb der lokalen
              Umgebung korrekt beschreiben.

            - **Keine Kausalität:** Ein hoher Feature-Beitrag bedeutet nicht,
              dass dieses Merkmal die Vorhersage kausal verursacht.
            """
        )
    st.info(
    """
    💡 **LIME ist nicht auf Tabellendaten beschränkt**

    LIME wurde als allgemeines, modellunabhängiges Erklärungsverfahren
    konzipiert. Das Grundprinzip kann auf verschiedene Datentypen
    angewendet werden.

    **Tabellendaten:** Einzelne Merkmale einer Instanz werden perturbiert.
    In unserem Beispiel sind dies z.B. Schnabellänge, Schnabeltiefe,
    Flossenlänge und Körpermasse.

    **Text:** Wörter bzw. Wortbestandteile können entfernt oder verändert
    werden. LIME untersucht dann, welche Wörter besonders stark zur
    Klassifikation eines Textes beitragen.

    **Bilder:** Bildbereiche bzw. Superpixel können verändert oder
    ausgeblendet werden. So lässt sich untersuchen, welche Bildregionen
    für eine Klassifikation besonders relevant sind.
    """
    )
    
    
    st.markdown("## Weiterführende Literatur")
    st.markdown(
    "- <b>Why Should I Trust You?: Explaining the Predictions of Any Classifier<b>, Ribeiro et al., 2016."
    )


#============================================================= SHAP
with tab_shap:
    st.markdown("## 5 · SHAP: SHapley Additive exPlanations")
    st.caption(
        "SHAP quantifiziert den fairen Beitrag jedes einzelnen Merkmals zur Modellentscheidung, basierend auf Konzepten der kooperativen Spieltheorie."
    )

    # ------------------------------------------------ 1. Erklärung & Formel
    st.markdown("### 1 · Erklärung der Methode: Spieltheorie & Mathematische Grundlagen")
    st.markdown(
        r"""
Die Methode **SHAP (SHapley Additive exPlanations)** überträgt ein klassisches Konzept der kooperativen Spieltheorie von Lloyd Shapley (1953, Wirtschaftsnobelpreis) auf das maschinelle Lernen:

- **Die Spieler:** Die einzelnen Merkmale $x_j$ (Schnabellänge, Schnabeltiefe, Flossenlänge, Körpergewicht) eines Pinguins.
- **Das Spiel:** Das trainierte Vorhersagemodell $f(x)$ (unser Random Forest).
- **Die Auszahlung (Payout):** Der Unterschied zwischen der konkreten Vorhersage $f(x)$ und der durchschnittlichen Baseline-Vorhersage $\mathbb{E}[f(X)]$.

Der **Shapley Value** $\phi_j$ weist jedem Merkmal $j$ einen fairen Anteil an diesem Gewinn zu. Er wird berechnet als der gewichtete mittlere **Marginalbeitrag** über alle möglichen Teilmengen (Koalitionen) $S$ der Merkmalsmenge $F$:

$$
\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \cdot \Big( v(S \cup \{j\}) - v(S) \Big)
$$

**Bedeutung der Komponenten in der Formel:**
- $F$: Die Menge aller Merkmale im Datensatz ($|F| = 4$).
- $S \subseteq F \setminus \{j\}$: Eine Koalition (Teilmenge) von Merkmalen, die Merkmal $j$ noch *nicht* enthält.
- $\frac{|S|! \, (|F| - |S| - 1)!}{|F|!}$: Das kombinatorische Gewicht – die Wahrscheinlichkeit, dass Merkmal $j$ bei einer zufälligen Reihenfolgebildung genau nach der Koalition $S$ hinzugefügt wird.
- $v(S \cup \{j\}) - v(S)$: Der **marginale Mehrwert**, den Merkmal $j$ zur bestehenden Koalition $S$ beisteuert (wobei $v(S)$ der Modell-Erwartungswert ist, wenn nur die Merkmale in $S$ bekannt sind).
"""
    )

    merkkasten(
        "Die vier mathematischen Garantien (Axiome) von SHAP",
        "• <b>1. Effizienz (Additivität):</b> $\\sum_{j=1}^{|F|} \\phi_j(x) = f(x) - \\mathbb{E}[f(X)]$ — die Summe aller Feature-Beiträge ergibt exakt die Abweichung vom globalen Durchschnitt.<br>"
        "• <b>2. Symmetrie:</b> Zwei Merkmale mit identischem Einfluss auf alle Koalitionen erhalten immer denselben Shapley Value.<br>"
        "• <b>3. Dummy (Null-Effekt):</b> Ein Merkmal, das den Vorhersagewert in keiner Koalition ändert, erhält exakt $\\phi_j = 0$.<br>"
        "• <b>4. TreeSHAP:</b> Für baumbasierte Ensembles (wie Random Forests) wertet TreeSHAP diese Summe nicht exponentiell ($2^{|F|}$), sondern in polynomieller Laufzeit $\\mathcal{O}(TLD^2)$ exakt entlang der Baumstrukturen aus.",
        typ="definition",
    )

    # ------------------------------------------------ Modell & Caching
    @st.cache_resource
    def load_shap_model_and_data():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import shap

        df = analyse.load_data()
        feature_names = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        X = df[feature_names]
        Y = df["species"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.35,
            random_state=42,
            stratify=Y,
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        classes = list(model.classes_)

        return model, explainer, shap_values, X_test, Y_test, classes, feature_names

    rf_model, tree_explainer, shap_vals, X_test, Y_test, class_names, feat_names = load_shap_model_and_data()

    # ------------------------------------------------ 2. Globale Ergebnisse
    st.markdown("---")
    st.markdown("### 2 · Interaktive globale Ergebnisse: Merkmalseinfluss pro Spezies")
    st.markdown(
        "Wähle eine Pinguinart aus. Anschließend werden die globalen Feature-Contributions (Beeswarm Plot & mittlere Wichtigkeit) für diese Zielklasse berechnet und visualisiert:"
    )

    selected_species = st.radio(
        "Ziel-Spezies auswählen:",
        options=class_names,
        index=2,  # Default: Gentoo
        horizontal=True,
    )
    species_idx = class_names.index(selected_species)

    import matplotlib.pyplot as plt
    import shap

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"**SHAP Beeswarm Plot ({selected_species})**")
        st.caption("Punkte = Pinguine im Testset. Farbe = Feature-Wert (rot = hoch, blau = niedrig). Position = Einfluss auf die Klassifikation.")
        fig_bee, ax_bee = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.beeswarm(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Einfluss der Features auf '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

    with col_g2:
        st.markdown(f"**Globale Feature Importance |SHAP| ({selected_species})**")
        st.caption("Mittlere absolute SHAP-Werte $\\frac{1}{n}\\sum |\\phi_j|$ – zeigt, welche Merkmale die stärkste Gesamthebelwirkung besitzen.")
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.bar(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Mittlere Wichtigkeit für '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # Spezifische Interpretation je nach ausgewählter Spezies
    if selected_species == "Gentoo":
        st.info(
            "🐧 **Erkenntnis für Gentoo:** Eine hohe Flossenlänge (`flipper_length_mm`) und eine hohe Körpermasse (`body_mass_g`) treiben die Modellvorhersage stark in Richtung Gentoo (rote Punkte weit rechts). Eine geringere Schnabeltiefe begünstigt Gentoo zusätzlich."
        )
    elif selected_species == "Chinstrap":
        st.info(
            "🐧 **Erkenntnis für Chinstrap (Zügelpinguin):** Eine lange Schnabellänge (`bill_length_mm`) in Kombination mit geringerer Körpermasse und moderaten Flossenlängen trennt Chinstrap sehr präzise von Adelie und Gentoo."
        )
    else:  # Adelie
        st.info(
            "🐧 **Erkenntnis für Adelie:** Kürzere Schnäbel (`bill_length_mm`) und kürzere Flossen sind das zentrale Erkennungsmerkmal für Adeliepinguine. Hohe Schnabellängen senken die Adelie-Wahrscheinlichkeit drastisch."
        )

    # ------------------------------------------------ 3. Lokale Ergebnisse (Vergleich LIME)
    st.markdown("---")
    st.markdown("### 3 · Interaktive lokale Ergebnisse: Einzelfall-Erklärung (Waterfall Plot)")
    st.markdown(
        "Hier untersuchen wir konkrete Pinguine aus dem Testdatensatz. Wähle einen Test-Pinguin aus, um den exakten Zerlegungspfad der Entscheidung nachzuvollziehen:"
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        preset_choice = st.selectbox(
            "Pinguin-Instanz auswählen:",
            options=[
                "Pinguin #4 (Tatsächlich: Gentoo) — Hohe Modellsicherheit",
                "Pinguin #0 (Tatsächlich: Gentoo) — Typisches Exemplar",
                "Pinguin #2 (Tatsächlich: Chinstrap) — Zügelpinguin",
                "Pinguin #7 (Tatsächlich: Adelie) — Adeliepinguin",
                "Pinguin #10 (Tatsächlich: Adelie) — Typisches Exemplar",
                "Eigene Index-Eingabe...",
            ],
            index=0,
        )
    with col_sel2:
        if preset_choice == "Eigene Index-Eingabe...":
            instance_idx = int(
                st.number_input(
                    "Test-Index (0 bis 116):",
                    min_value=0,
                    max_value=len(X_test) - 1,
                    value=4,
                )
            )
        else:
            preset_map = {
                "Pinguin #4 (Tatsächlich: Gentoo) — Hohe Modellsicherheit": 4,
                "Pinguin #0 (Tatsächlich: Gentoo) — Typisches Exemplar": 0,
                "Pinguin #2 (Tatsächlich: Chinstrap) — Zügelpinguin": 2,
                "Pinguin #7 (Tatsächlich: Adelie) — Adeliepinguin": 7,
                "Pinguin #10 (Tatsächlich: Adelie) — Typisches Exemplar": 10,
            }
            instance_idx = preset_map[preset_choice]

    # Daten des ausgewählten Pinguins
    p_instance = X_test.iloc[[instance_idx]]
    p_true = Y_test.iloc[instance_idx]
    p_pred = rf_model.predict(p_instance)[0]
    p_pred_idx = class_names.index(p_pred)
    p_proba = rf_model.predict_proba(p_instance)[0]

    # Datenübersicht anzeigen
    st.markdown("**Merkmalswerte und Modellprognose für diesen Pinguin:**")
    metrics_cols = st.columns(6)
    metrics_cols[0].metric("Schnabellänge", f"{p_instance['bill_length_mm'].values[0]:.1f} mm")
    metrics_cols[1].metric("Schnabeltiefe", f"{p_instance['bill_depth_mm'].values[0]:.1f} mm")
    metrics_cols[2].metric("Flossenlänge", f"{p_instance['flipper_length_mm'].values[0]:.0f} mm")
    metrics_cols[3].metric("Körpermasse", f"{p_instance['body_mass_g'].values[0]:.0f} g")
    metrics_cols[4].metric("Tatsächlich", p_true)
    metrics_cols[5].metric("Vorhersage", f"{p_pred} ({p_proba[p_pred_idx]:.1%})")

    # Waterfall Plot für die vorhergesagte Klasse
    fig_water, ax_water = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_vals[instance_idx, :, p_pred_idx], show=False)
    plt.title(f"SHAP Waterfall Plot für Pinguin #{instance_idx} (Erklärung für Vorhersage: '{p_pred}')", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig_water, use_container_width=True)
    plt.close(fig_water)

    st.markdown(
        f"""
**Interpretation des Waterfall Plots im Vergleich zu LIME (Tab 4):**
- Der Plot startet unten beim Erwartungswert $\\mathbb{{E}}[f(X)] = {shap_vals[instance_idx, :, p_pred_idx].base_values:.2f}$ (Basis-Wahrscheinlichkeit vor Kenntnis der Merkmale).
- Jeder rote Balken $(+)$ erhöht die Wahrscheinlichkeit für **{p_pred}**, jeder blaue Balken $(-)$ senkt sie.
- Am Ende summiert sich alles exakt auf den finalen Ausgabewert $f(x) = {p_proba[p_pred_idx]:.2f}$.
- **Unterschied zu LIME:** Während LIME eine *lokale Heuristik* über Zufallsstörungen trainiert, liefert SHAP eine *exakte, spieltheoretisch garantierte* additive Attribution.
"""
    )

    # ------------------------------------------------ 4. Limitationen
    st.markdown("---")
    st.markdown("### 4 · Worin liegen die Limitations von SHAP?")
    merkkasten(
        "Kritische Grenzen und Fallstricke von SHAP",
        "1. <b>Erklärung des Modells $\\neq$ Erklärung der Wirklichkeit (Keine Kausalität):</b> SHAP erklärt ausschließlich die interne mathematische Funktionsweise des trainierten Modells. Wenn das Modell einen Scheinzusammenhang oder Confounder lernt, erhält dieser einen hohen SHAP-Wert, obwohl in der realen Biologie kein Kausaleffekt vorliegt.<br><br>"
        "2. <b>Annahme unabhängiger Merkmale:</b> Standard-KernelSHAP permutiert Features unabhängig. Bei korrelierten Merkmalen (z. B. Flossenlänge und Körpergewicht) werden Koalitionen evaluiert, die in der Realität physikalisch unmöglich sind (z. B. 6 kg Pinguin mit 130 mm Flossen).<br><br>"
        "3. <b>Rechenaufwand bei modellagnostischer Nutzung:</b> Für beliebige Blackbox-Modelle müssen bei $M$ Features potenziell alle $2^M$ Koalitionen berechnet werden. TreeSHAP umgeht dies für Entscheidungsbäume, ist jedoch an diese Modellklasse gebunden.<br><br>"
        "4. <b>Multi-Class-Komplexität:</b> Da SHAP für jede Klasse separate Attributionswerte berechnet, kann die gleichzeitige Interpretation von 3 oder mehr Klassen für Anwender kognitiv anspruchsvoll sein.",
        typ="achtung",
    )

    # ------------------------------------------------ 5. Weiterführende Literatur
    st.markdown("---")
    st.markdown("### 5 · Weiterführende Literatur")
    st.markdown(
        """
- **Lundberg, S. M., & Lee, S.-I. (2017):** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017), 4765–4774. *(Das Grundlagen-Paper zu SHAP)*
- **Lundberg, S. M. et al. (2020):** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56–67. *(TreeSHAP-Algorithmus)*
- **Shapley, L. S. (1953):** *A Value for n-Person Games.* Contributions to the Theory of Games, 2(28), 307–317. *(Originale spieltheoretische Formulierung)*
- **Molnar, C. (2022):** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* Kapitel 9.5 (Shapley Values) und Kapitel 9.6 (SHAP).
- **Slack, D. et al. (2020):** *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AIES 2020.
"""
    )


#============================================================= Ausblick
with tab_ausblick:
    st.markdown("## 6 · Ausblick: Andere Ansätze der Explainable AI")
    st.caption(
        "Weitere globale und lokale Methoden zur Interpretation von KI-Modellen."
    )