""" Explainable AI App
Frage: Wie koennen wir die Art von Pinguinen erkennen 
und die Entscheidung des KI-Modells nachvollziehen?
"""

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

# ------------------------------------------------ Daten & Modell laden
data = analyse.load_data()
model, X_test, y_test, y_pred = analyse.train_blackbox(data, report=False)

# ------------------------------------------------ Header & Einleitung
st.markdown("# Explainable AI: Wie können wir Black-Box Modelle verstehen?")
st.caption(
    "Projekt von Katharina Gudat und Leon Kraus · Daten: Palmer Penguins, 2014, 344 Pinguine"
)

merkkasten(
    "Projektübersicht: Entscheidungen von KI-Modellen transparent machen",
    "Dieses interaktive Projekt veranschaulicht Methoden der <b>Explainable AI (XAI)</b> am Beispiel der Artklassifikation von antarktischen Pinguinen.<br><br>"
    "• <b>Tab 1:</b> Einleitung in die Konzepte und Taxonomie von xAI<br>"
    "• <b>Tab 2:</b> Erkundung des <i>Palmer Penguins</i> Datensatzes<br>"
    "• <b>3:</b> Training und Evaluation des Random-Forest Black-Box-Modells<br>"
    "• <b>4 & 5:</b> Detaillierte praktische Untersuchung der Methoden <b>LIME</b> und <b>SHAP</b><br>"
    "• <b>Tab 6:</b> Fazit, Gegenüberstellung (By-Design vs. Post-Hoc) und Ausblick auf moderne Forschungsansätze",
    typ="merke",
)

tab_intro, tab_daten, tab_blackbox, tab_lime, tab_shap, tab_ausblick = st.tabs(
    [
        "1 · Einleitung",
        "2 · Daten: Palmer Penguins",
        "3 · Das Blackbox-Modell",
        "4 · LIME",
        "5 · SHAP",
        "6 · Ausblick",
    ]
)

# ============================================================= Tab 1: Einleitung
with tab_intro:
    st.markdown("## 1 · Einleitung: Explainable AI")
    st.markdown(
        "**Explainable AI (xAI)** ist ein zentrales Forschungsgebiet des modernen maschinellen Lernens. "
        "Es befasst sich mit der Frage, wie die internen Entscheidungsprozesse komplexer Modelle für Menschen verständlich, "
        "plausibel und überprüfbar gemacht werden können."
    )
    st.markdown(
        "Die folgende Übersicht zeigt die grundlegende Taxonomie und Einteilung gängiger xAI-Verfahren:"
    )

    image_path = os.path.join(ORDNER, "images", "xAI_overview.png")
    if os.path.exists(image_path):
        st.image(image_path, caption="Abbildung 1: Überblick über die Taxonomie von Explainable AI Methoden", use_container_width=True)

    st.markdown(
        """
        ### Die zwei Hauptachsen der Interpretierbarkeit

        1. **By Design (Intrinsisch) vs. Post-Hoc (Nachträglich):**
           - **By Design:** Einfache Modelle (wie lineare Regression oder flache Entscheidungsbäume) sind durch ihre Bauart direkt nachvollziehbar (*erklärbar by design*).
           - **Post-Hoc:** Bei komplexen Modellen (wie Random Forests, Gradient Boosted Trees oder tiefen neuronalen Netzen) wird das Erklärungsverfahren erst *nach* dem Training von außen auf das fertige Modell angewendet.

        2. **Lokal vs. Global:**
           - **Lokale Methoden:** Erklären eine einzelne, konkrete Vorhersage für ein spezifisches Individuum (z. B. *„Warum ist Pinguin #7 ein Gentoo?“*).
           - **Globale Methoden:** Beschreiben das Gesamtverhalten des Modells über den gesamten Merkmalsraum hinweg.

        3. **Modellagnostisch vs. Modellspezifisch:**
           - **Modellagnostisch:** Funktioniert mit jedem beliebigen Machine-Learning-Modell (z. B. LIME und KernelSHAP).
           - **Modellspezifisch:** Nutzt interne mathematische Eigenschaften bestimmter Architekturen aus (z. B. TreeSHAP für Baum-Ensembles oder Grad-CAM für Convolutional Neural Networks).
        """
    )

    st.markdown(
        """
        ### Warum brauchen wir Erklärbarkeit?

        In der Praxis reicht eine hohe Trefferquote (Accuracy) alleine selten aus:
        - **Vertrauen & Validierung:** Trifft das Modell Entscheidungen anhand biologisch plausibler Merkmale oder verlässt es sich auf Artefakte und Scheinkorrelationen?
        - **Fehleranalyse & Debugging:** Warum wurde ein bestimmtes Testexemplar falsch klassifiziert? Welches Merkmal hat den Ausschlag gegeben?
        - **Verantwortung & Regulierung:** In kritischen Anwendungsfeldern (Medizin, Justiz, Finanzen) fordern Nutzer und Regulatoren nachvollziehbare Begründungen.
        """
    )

    merkkasten(
        "Fokus dieses Projekts: Lokale modellagnostische Post-Hoc Verfahren",
        "In den Tabs 4 und 5 untersuchen wir die zwei führenden Verfahren im praktischen Einsatz:<br>"
        "• <b>LIME (Tab 4):</b> Erklärt einzelne Vorhersagen lokal durch ein einfaches lineares Ersatzmodell in der direkten Nachbarschaft eines Datenpunkts.<br>"
        "• <b>SHAP (Tab 5):</b> Nutzt Konzepte der kooperativen Spieltheorie (Shapley Values), um den exakten, mathematisch fairen Beitrag jedes einzelnen Merkmals zu bestimmen.",
        typ="definition",
    )


# ============================================================= Tab 2: Daten
with tab_daten:
    st.markdown("## 2 · Daten: Palmer Penguins")
    st.caption(
        "Datensatz von Dr. Kristen Gorman und der Palmer Station LTER (Long Term Ecological Research), 2014."
    )
    st.markdown(
        "Der Datensatz enthält Messungen von 344 Pinguinen aus der Antarktis, verteilt auf drei Arten (*Adelie*, *Chinstrap*, *Gentoo*) "
        "und drei Inseln des Palmer-Archipels (*Biscoe*, *Dream*, *Torgersen*)."
    )

    st.dataframe(data.sample(7, random_state=42), hide_index=True, use_container_width=True)

    st.markdown(
        "Für unser Klassifikationsmodell verwenden wir die vier kontinuierlichen Körpermessungen als **Eingangsmerkmale (Features)**:"
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Feature 1", "Schnabellänge", "bill_length_mm")
    col_m2.metric("Feature 2", "Schnabeltiefe", "bill_depth_mm")
    col_m3.metric("Feature 3", "Flossenlänge", "flipper_length_mm")
    col_m4.metric("Feature 4", "Körpermasse", "body_mass_g")

    st.markdown("---")
    st.markdown("### Interaktive Merkmals-Erkundung")
    st.markdown(
        "Wähle zwei Merkmale aus, um die Trennbarkeit der Pinguinarten im Streudiagramm visuell zu überprüfen:"
    )

    numeric_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        x_feature = st.selectbox("X-Achse:", numeric_features, index=0)
    with col_f2:
        y_feature = st.selectbox("Y-Achse:", numeric_features, index=3)

    fig_scatter = px.scatter(
        data,
        x=x_feature,
        y=y_feature,
        color="species",
        hover_data=["island", "sex"],
        title=f"Verteilung der Pinguinarten: {x_feature} vs. {y_feature}",
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        "**Beobachtung:** Einige Arten lassen sich bereits anhand von zwei Merkmalen recht gut abgrenzen (z. B. Gentoo durch lange Flossen und hohe Körpermasse). "
        "Bei Adelie und Chinstrap überschneiden sich einzelne Merkmale jedoch, weshalb ein mehrdimensionales Klassifikationsmodell notwendig ist."
    )


# ============================================================= Tab 3: Blackbox-Modell
with tab_blackbox:
    st.markdown("## 3 · Das Blackbox-Modell")
    st.markdown(
        "Wir trainieren einen **Random Forest Classifier** mit 70 Entscheidungsbäumen auf 70 % der Daten, "
        "um die Spezies anhand der vier numerischen Merkmale vorherzusagen. Die restlichen 30 % dienen als Testdatensatz."
    )

    accuracy = accuracy_score(y_test, y_pred)

    col_acc, col_info = st.columns([1, 3])
    with col_acc:
        st.metric("Test-Genauigkeit (Accuracy)", f"{accuracy:.1%}")
    with col_info:
        st.markdown(
            "Das Modell erzielt eine sehr hohe Treffsicherheit auf den ungesehenen Testdaten. "
            "Da ein Random Forest jedoch aus vielen verzweigten Einzelbäumen besteht, die per Mehrheitsentscheid abstimmen, "
            "ist die Entscheidung für einen einzelnen Pinguin **nicht mehr unmittelbar durch einfaches Hinsehen ablesbar** (Black Box)."
        )

    st.markdown("---")
    st.markdown("### Modell-Evaluation: Confusion Matrix & Globale Feature Importance")

    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.markdown("**Confusion Matrix (Testdatensatz)**")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig_cm = px.imshow(
            cm,
            x=model.classes_,
            y=model.classes_,
            text_auto=True,
            labels={"x": "Vorhergesagte Art", "y": "Tatsächliche Art", "color": "Anzahl"},
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Die Matrix zeigt, dass nur sehr wenige Fehlklassifikationen (hauptsächlich zwischen Adelie und Chinstrap) auftreten.")

    with col_eval2:
        st.markdown("**Globale Gini Feature Importance**")
        importance_df = pd.DataFrame({
            "Feature": analyse.standard_features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Globale Feature Importance des Random Forests",
            labels={"Importance": "Mittlere Gini-Wichtigkeit", "Feature": "Merkmal"},
        )
        fig_importance.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_importance.update_layout(
            xaxis_range=[0, importance_df["Importance"].max() * 1.2],
            height=380,
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption("Gibt an, wie oft und wie stark ein Merkmal über alle Bäume hinweg zur Trennung der Klassen beiträgt.")

    st.markdown(
        """
        ### Die Grenze globaler Metriken
        Die Gini Feature Importance sagt uns zwar, dass die **Schnabellänge** und **Flossenlänge** global die einflussreichsten Merkmale sind.
        Sie kann uns jedoch **nicht** beantworten:
        - *Warum wurde ein bestimmter Pinguin als Adelie eingestuft?*
        - *Welches Merkmal hat bei einem konkreten Einzelfall gegen eine andere Klasse gesprochen?*

        Genau hierfür benötigen wir lokale Post-Hoc-Erklärungsverfahren wie **LIME (Tab 4)** und **SHAP (Tab 5)**.
        """
    )


# ============================================================= Tab 4: LIME
with tab_lime:
    st.markdown("## 4 · LIME: Local Interpretable Model-agnostic Explanations")
    st.caption(
        "LIME erklärt einzelne Modellvorhersagen lokal durch ein interpretierbares Ersatzmodell in der direkten Umgebung des Datenpunkts."
    )

    st.markdown(
        """
        ### Die Grundidee von LIME
        Auch wenn ein hochdimensionales Machine-Learning-Modell global hochgradig nicht-linear und komplex ist, 
        verhält es sich in der **unmittelbaren lokalen Nachbarschaft** eines einzelnen Datenpunkts näherungsweise linear.
        
        LIME erzeugt künstliche Perturbationen (Zufallsstörungen) um die ausgewählte Instanz herum, fragt das Black-Box-Modell nach dessen Vorhersagen für diese Punkte 
        und trainiert anschließend ein einfaches, gewichtetes Surrogatmodell (z. B. eine lineare Regression).
        """
    )

    st.markdown("### Mathematische Grundlagen")
    st.markdown(
        r"""
        Für eine Instanz $x$ approximiert LIME das Black-Box-Modell $f$ durch ein lineares Erklärungsmodell $g$:
        """
    )
    st.latex(r"g(x') = \beta_0 + \sum_{j=1}^{p} \beta_j x'_j")
    st.markdown(
        r"""
        - $x'$ sind die Merkmale einer perturbierten Instanz.
        - $\beta_j$ quantifiziert den **lokalen Einfluss** des Merkmals $j$:
          - Ein **positiver Koeffizient** $(\beta_j > 0)$ stützt die betrachtete Klasse.
          - Ein **negativer Koeffizient** $(\beta_j < 0)$ spricht gegen diese Klasse.
        """
    )

    st.markdown("#### Optimierungsproblem & Kernel-Gewichtung")
    st.markdown(
        r"""
        LIME minimiert den Fehler der lokalen Approximation bei gleichzeitiger Begrenzung der Modellkomplexität $\Omega(g)$:
        """
    )
    st.latex(r"\xi(x) = \underset{g \in G}{\operatorname{arg\,min}} \Big[ L(f, g, \pi_x) + \Omega(g) \Big]")
    st.markdown(
        r"""
        Die Nähe eines gestörten Datenpunkts $x'$ zur Originalinstanz $x$ wird über einen exponentiellen Distanz-Kernel gewichtet:
        """
    )
    st.latex(r"\pi_x(x') = \exp\left( -\frac{D(x, x')^2}{\sigma^2} \right)")
    st.markdown(
        r"""
        wobei $\sigma$ die **Kernelbreite** (*kernel width*) ist:
        - **Kleine Kernelbreite:** Nur Datenpunkte in unmittelbarer Nähe erhalten Gewicht $\rightarrow$ stark lokale, sensitive Erklärung.
        - **Große Kernelbreite:** Auch weiter entfernte Punkte fließen ein $\rightarrow$ glattere, globalere Näherung.
        """
    )

    st.markdown("---")
    st.markdown("### Interaktive LIME-Erklärung für Test-Pinguine")

    if "lime_instance_idx" not in st.session_state:
        st.session_state["lime_instance_idx"] = 7

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

    col_prev, col_num, col_next = st.columns([1, 2, 1], vertical_alignment="center")

    with col_prev:
        st.button("◀ Vorheriger", key="lime_btn_prev", on_click=_prev_penguin, disabled=(st.session_state["lime_instance_idx"] <= 0), use_container_width=True)

    with col_num:
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<span style='font-size: 1.6rem; font-weight: 700;'>Pinguin #{st.session_state['lime_instance_idx']}</span>"
            f"<br><span style='font-size: 0.85rem; color: #888;'>Index {st.session_state['lime_instance_idx']} von {len(X_test) - 1}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        st.button("Nächster ▶", key="lime_btn_next", on_click=_next_penguin, disabled=(st.session_state["lime_instance_idx"] >= len(X_test) - 1), use_container_width=True)

    instance_idx = st.session_state["lime_instance_idx"]
    instance_to_explain = X_test.iloc[instance_idx].to_numpy()
    actual_class = y_test.iloc[instance_idx]
    predicted_class = model.predict(instance_to_explain.reshape(1, -1))[0]

    st.write(f"**Tatsächliche Art:** `{actual_class}` &nbsp;|&nbsp; **Vorhergesagte Art:** `{predicted_class}`")
    st.dataframe(X_test.iloc[[instance_idx]], hide_index=True, use_container_width=True)

    kernel_width = st.slider(
        "Kernelbreite (Kernel Width) — Wie lokal soll die Umgebung sein?",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    explainer = analyse.create_lime_explainer(model, data, kernel_width)
    explanation = analyse.explain_instance(explainer, model, instance_to_explain)
    lime_values = explanation.as_list(explanation.top_labels[0])

    lime_df = pd.DataFrame(lime_values, columns=["Feature", "Beitrag zur Vorhersage"]).sort_values("Beitrag zur Vorhersage", ascending=True)

    fig_lime = px.bar(
        lime_df,
        x="Beitrag zur Vorhersage",
        y="Feature",
        orientation="h",
        title=f"LIME Erklärung für Vorhersage: {predicted_class} (Kernelbreite = {kernel_width})",
        labels={"Beitrag zur Vorhersage": "Lokaler Beitrag (Koeffizient)", "Feature": "Merkmal"},
        color="Beitrag zur Vorhersage",
        color_continuous_scale=["#3E6DB5", "#E8804C"],
    )
    fig_lime.add_vline(x=0, line_width=2, line_color="black")
    fig_lime.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_lime, use_container_width=True)

    st.markdown(
        "**Erkenntnis:** Verändere die Kernelbreite im Schieberegler oben. Bei sehr kleinen Werten sinken die Beiträge stark ab; "
        "bei größeren Werten stabilisiert sich die Erklärung, verliert aber etwas an lokaler Schärfe."
    )

    st.markdown("---")
    st.markdown("### Stärken und Schwächen von LIME")

    col_l_s, col_l_w = st.columns(2)
    with col_l_s:
        st.success(
            """
            **✓ Stärken von LIME**
            - **Modellagnostisch:** Funktioniert mit jedem Klassifikator ohne Zugriff auf interne Gewichte.
            - **Lokale Präzision:** Erklärt spezifische Einzelfall-Entscheidungen sehr anschaulich.
            - **Vielseitig:** Über Tabellendaten hinaus auch für Bild- (Superpixel) und Textdaten (Wort-Entfernung) anwendbar.
            """
        )
    with col_l_w:
        st.warning(
            """
            **⚠ Schwächen von LIME**
            - **Hyperparameter-Sensitivität:** Die Erklärung hängt stark von der gewählten Kernelbreite ab.
            - **Stochastisch:** Durch Zufallssampling können Erklärungen zwischen Läufen leicht variieren.
            - **Keine Additivitäts-Garantie:** Die Beiträge summieren sich nicht exakt auf die Modellwahrscheinlichkeit auf.
            """
        )

    st.info(
        """
        💡 **Wichtig: Erklärung ≠ Kausalität**  
        Die Feature-Beiträge von LIME beschreiben, wie das trainierte Modell seine Eingaben gewichtet.
        Sie spiegeln **keine kausalen Zusammenhänge** in der realen Biologie wider und sind **keine Kontrafaktischen Aussagen**
        (ein positiver Beitrag bedeutet nicht, dass eine künstliche Gewichtsänderung die Art des Pinguins ändern würde).
        """
    )

    st.markdown("---")
    


# ============================================================= Tab 5: SHAP
with tab_shap:
    st.markdown("## 5 · SHAP: Shapley Additive Explanations")
    st.caption(
        "SHAP basiert auf der kooperativen Spieltheorie und quantifiziert den mathematisch fairen Beitrag jedes Merkmals."
    )

    st.markdown(
        r"""
        ### Grundlagen: Spieltheorie & Mathematische Theorie

        Die Methode **SHAP (Shapley Additive Explanations)** überträgt ein klassisches Konzept der kooperativen Spieltheorie von Lloyd Shapley (1953, Wirtschaftsnobelpreis) auf das maschinelle Lernen:
        - **Die Spieler:** Die einzelnen Merkmale $x_j$ (Schnabellänge, Schnabeltiefe, Flossenlänge, Körpermasse) eines Pinguins.
        - **Das Spiel:** Das trainierte Modell $f(x)$ (unser Random Forest).
        - **Die Auszahlung (Payout):** Die Differenz zwischen der Vorhersage $f(x)$ und der durchschnittlichen Basis-Vorhersage $\mathbb{E}[f(X)]$.

        Der **Shapley Value** $\phi_j$ weist jedem Merkmal $j$ einen fairen Anteil an diesem Gewinn zu:
        """
    )
    st.latex(r"\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \cdot \Big( v(S \cup \{j\}) - v(S) \Big)")
    st.markdown(
        r"""
        - $F$: Menge aller Merkmale ($|F| = 4$).
        - $S$: Koalition (Teilmenge) von Merkmalen ohne Merkmal $j$.
        - $v(S \cup \{j\}) - v(S)$: Der **marginale Mehrwert**, den Merkmal $j$ zur Koalition $S$ beisteuert.
        """
    )

    merkkasten(
        "Die vier mathematischen Axiome von SHAP",
        "• <b>1. Effizienz (Additivität):</b> ∑ φ<sub><i>j</i></sub>(<i>x</i>) = <i>f</i>(<i>x</i>) − 𝔼[<i>f</i>(<i>X</i>)] — alle Feature-Beiträge summieren sich exakt auf die Abweichung vom globalen Durchschnitt.<br>"
        "• <b>2. Symmetrie:</b> Zwei Merkmale mit identischem Beitrag in allen Koalitionen erhalten stets denselben Shapley Value.<br>"
        "• <b>3. Dummy (Null-Effekt):</b> Ein Merkmal ohne Einfluss auf den Vorhersagewert erhält exakt φ<sub><i>j</i></sub> = 0.<br>"
        "• <b>4. TreeSHAP:</b> Für baumbasierte Modelle wertet TreeSHAP diese Summe in polynomieller Laufzeit <i>O</i>(<i>T · L · D</i>²) exakt entlang der Baumstrukturen aus.",
        typ="definition",
    )

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

        X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
            X, Y, test_size=0.35, random_state=42, stratify=Y
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_s, Y_train_s)

        exp = shap.TreeExplainer(rf)
        sv = exp(X_test_s)
        cls = list(rf.classes_)

        return rf, exp, sv, X_test_s, Y_test_s, cls, feature_names

    rf_model, tree_explainer, shap_vals, X_test_s, Y_test_s, class_names, feat_names = load_shap_model_and_data()

    st.markdown("---")
    st.markdown("### Globale Ergebnisse: Merkmalseinfluss pro Spezies")
    st.markdown(
        "Wähle eine Pinguinart aus, um die globalen SHAP-Attributionen für diese Zielklasse zu untersuchen:"
    )

    selected_species = st.radio("Ziel-Spezies auswählen:", options=class_names, index=2, horizontal=True)
    species_idx = class_names.index(selected_species)

    import matplotlib.pyplot as plt
    import shap

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"**SHAP Beeswarm Plot ({selected_species})**")
        st.caption("Punkte = Pinguine; Farbe = Merkmalswert (rot = hoch, blau = niedrig); X-Position = Einfluss")
        fig_bee, ax_bee = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.beeswarm(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Einfluss der Merkmale auf '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

    with col_g2:
        st.markdown(f"**Mittlere absolute Wichtigkeit |SHAP| ({selected_species})**")
        st.caption("Mittlerer Hebel $\\frac{1}{n}\\sum |\\phi_j|$ des jeweiligen Merkmals")
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.bar(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Mittlere Wichtigkeit für '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    if selected_species == "Gentoo":
        st.info("🐧 **Erkenntnis für Gentoo:** Hohe Flossenlängen (`flipper_length_mm`) und hohes Körpergewicht (`body_mass_g`) treiben die Vorhersage stark zu Gentoo (rote Punkte rechts). Geringe Schnabeltiefe begünstigt Gentoo zusätzlich.")
    elif selected_species == "Chinstrap":
        st.info("🐧 **Erkenntnis für Chinstrap:** Lange Schnäbel (`bill_length_mm`) treiben die Vorhersage stark zu Chinstrap. Geringere Körpermasse, kürzere Flossen und hohe Schnabeltiefe stützen Chinstrap.")
    else:  # Adelie
        st.info("🐧 **Erkenntnis für Adelie:** Kurze Schnäbel (`bill_length_mm`) und kurze Flossen (`flipper_length_mm`) treiben die Entscheidung stark zu Adelie (blaue Punkte rechts). Hohe Schnabeltiefe begünstigt Adelie zusätzlich.")

    st.markdown("---")
    st.markdown("### Lokale Ergebnisse: Einzelfall-Erklärung (Waterfall Plot)")
    st.markdown("Wähle einen Test-Pinguin aus, um die schrittweise Zusammensetzung der Vorhersage nachzuvollziehen:")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        preset_choice = st.selectbox(
            "Pinguin-Instanz auswählen:",
            options=[
                "Pinguin #4 (Gentoo)",
                "Pinguin #0 (Gentoo)",
                "Pinguin #2 (Chinstrap)",
                "Pinguin #59 (Chinstrap)",
                "Pinguin #10 (Adelie)",
                "Pinguin #20 (Adelie — Fehlklassifikation)",
                "Eigene Index-Eingabe...",
            ],
            index=0,
        )
    with col_sel2:
        if preset_choice == "Eigene Index-Eingabe...":
            inst_shap_idx = int(st.number_input("Test-Index (0 bis 116):", min_value=0, max_value=len(X_test_s) - 1, value=4))
        else:
            preset_map = {
                "Pinguin #4 (Gentoo)": 4,
                "Pinguin #0 (Gentoo)": 0,
                "Pinguin #2 (Chinstrap)": 2,
                "Pinguin #59 (Chinstrap)": 59,
                "Pinguin #10 (Adelie)": 10,
                "Pinguin #20 (Adelie — Fehlklassifikation)": 20,
            }
            inst_shap_idx = preset_map[preset_choice]

    p_instance = X_test_s.iloc[[inst_shap_idx]]
    p_true = Y_test_s.iloc[inst_shap_idx]
    p_pred = rf_model.predict(p_instance)[0]
    p_pred_idx = class_names.index(p_pred)
    p_proba = rf_model.predict_proba(p_instance)[0]

    st.markdown("**Merkmalswerte und Modellprognose:**")
    m_cols = st.columns(6)
    m_cols[0].metric("Schnabellänge", f"{p_instance['bill_length_mm'].values[0]:.1f} mm")
    m_cols[1].metric("Schnabeltiefe", f"{p_instance['bill_depth_mm'].values[0]:.1f} mm")
    m_cols[2].metric("Flossenlänge", f"{p_instance['flipper_length_mm'].values[0]:.0f} mm")
    m_cols[3].metric("Körpermasse", f"{p_instance['body_mass_g'].values[0]:.0f} g")
    m_cols[4].metric("Tatsächlich", p_true)
    m_cols[5].metric("Vorhersage", f"{p_pred} ({p_proba[p_pred_idx]:.1%})")

    fig_water, ax_water = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_vals[inst_shap_idx, :, p_pred_idx], show=False)
    plt.title(f"SHAP Waterfall Plot für Pinguin #{inst_shap_idx} (Erklärung für: '{p_pred}')", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig_water, use_container_width=True)
    plt.close(fig_water)

    st.markdown(
        f"""
#### 💡 Der Unterschied zwischen SHAP und LIME einfach erklärt:
- **So liest du den Waterfall Plot:**
  - **Startwert (unten):** Die Basis-Wahrscheinlichkeit $\\mathbb{{E}}[f(X)] = {shap_vals[inst_shap_idx, :, p_pred_idx].base_values:.2f}$ (vor Kenntnis der Merkmale).
  - 🔴 **Rote Balken (+):** Erhöhen die Wahrscheinlichkeit für **{p_pred}**.
  - 🔵 **Blaue Balken (-):** Senken die Wahrscheinlichkeit.
  - **Endwert (oben):** Ergibt exakt die finale Vorhersagewahrscheinlichkeit $f(x) = {p_proba[p_pred_idx]:.2f}$ ({p_proba[p_pred_idx]:.1%}).

- **Vergleich LIME vs. SHAP:**
  - **LIME:** Verändert Werte zufällig in der Nachbarschaft $\\rightarrow$ *heuristische Näherung*, kann stochastisch schwanken.
  - **SHAP:** Berechnet faire Team-Beiträge aller Merkmale $\\rightarrow$ *mathematisch exakt und additiv* (keine Näherungslücken).
"""
    )

    st.markdown("---")
    st.markdown("### Limitationen von SHAP")
    merkkasten(
        "Kritische Grenzen und Fallstricke von SHAP",
        "1. <b>Erklärung des Modells ≠ Erklärung der Wirklichkeit:</b> SHAP erklärt die innere Funktionsweise des trainierten Modells, keine biologische Kausalität.<br><br>"
        "2. <b>Annahme unabhängiger Merkmale:</b> Bei korrelierten Merkmalen evaluiert KernelSHAP teils unrealistische Merkmalskombinationen.<br><br>"
        "3. <b>Rechenaufwand bei modellagnostischer Nutzung:</b> Für beliebige Black-Box-Modelle erfordert KernelSHAP potenziell exponentiellen Rechenaufwand 2<sup><i>M</i></sup>. TreeSHAP löst dies für Entscheidungsbäume exakt in polynomieller Zeit.<br><br>"
        "4. <b>Multi-Class-Komplexität:</b> Da SHAP separate Werte pro Klasse berechnet, erfordert die Interpretation mehrerer Klassen Sorgfalt.",
        typ="achtung",
    )

    st.markdown("---")
    


# ============================================================= Tab 6: Ausblick
with tab_ausblick:
    st.markdown("## 6 · Ausblick: Weitere Ansätze der Explainable AI")
    st.caption(
        "Fazit der Analysen und ein strukturierter Überblick über moderne Forschungsrichtungen im Bereich Explainable AI."
    )

    st.markdown("### Fazit: Was lernen wir aus LIME und SHAP?")
    st.markdown(
        """
        LIME und SHAP bieten zwei leistungsfähige Methoden, um die Entscheidungen komplexer Machine-Learning-Modelle transparent zu machen.
        Sie zeigen, welche Merkmale für eine Klassifikation den Ausschlag gegeben haben, und ermöglichen gezieltes Debugging sowie Validierung von Modellen.

        Gleichzeitig gilt: **Erklärungsverfahren sind Werkzeuge zur Modellprüfung – sie liefern keine absolute Wahrheit über die Realität.**
        Unterschiedliche Methoden betrachten das Modell aus unterschiedlichen Blickwinkeln. Für verlässliche Analysen empfiehlt sich stets der vergleichende Einsatz mehrerer Verfahren.
        """
    )

    st.info(
        """
        💡 **Kernaussage:** Eine XAI-Methode erklärt das mathematische Verhalten des Modells aus den Trainingsdaten und keine kausalen Naturgesetze.
        """
    )

    st.markdown("---")
    st.markdown("### Die zwei Grundparadigmen moderner XAI-Forschung")

    merkkasten(
        "By Design (Intrinsisch) vs. Post-Hoc (Nachträglich)",
        "• <b>1. Interpretierbarkeit 'By Design':</b> Das Modell ist von Grund auf so aufgebaut, dass jeder Rechenschritt für Menschen direkt nachvollziehbar ist.<br>"
        "• <b>2. Post-Hoc Erklärungen:</b> Ein beliebig komplexes Black-Box-Modell wird trainiert, und externe Verfahren analysieren im Nachhinein dessen Verhalten.",
        typ="definition",
    )

    st.markdown("### 1 · Forschungsansätze 'By Design' (Intrinsische Interpretierbarkeit)")
    st.markdown(
        "Der Leitgedanke lautet: *„Warum eine Black Box mühsam von außen approximieren, wenn man direkt ein hochpräzises, verständliches Modell bauen kann?“* *(Rudin, 2019)*."
    )

    col_bd1, col_bd2 = st.columns(2)
    with col_bd1:
        st.markdown("#### 🌲 Explainable Boosting Machines (EBM / GAMs)")
        st.markdown(
            """
            **Generalized Additive Models (GAMs)** zerlegen die Vorhersage in eine Summe separater 1D-Kurven:
            
            $$g(E[y]) = \\beta_0 + f_1(x_1) + f_2(x_2) + \\dots + f_{ij}(x_i, x_j)$$
            
            - **Vorteil:** Erreicht auf Tabellendaten oft die gleiche Trefferquote wie Random Forests oder XGBoost, bleibt aber als exakter Funktionsgraph lesbar.
            """
        )
    with col_bd2:
        st.markdown("#### 🧩 Konzept-Flaschenhals-Modelle (CBMs)")
        st.markdown(
            """
            - **Prinzip:** Tiefe Netze lernen zunächst verständliche Zwischenkonzepte (z. B. *„Schnabel gebogen?“*, *„Gelbe Federn?“*).
            - **Entscheidung:** Erst daraus trifft ein einfaches lineares Modell die Endklassifikation.
            - **Intervention:** Fachexperten können bei Fehlentscheidungen direkt an den Zwischenkonzepten eingreifen.
            """
        )

    st.markdown("---")
    st.markdown("### 2 · Fortgeschrittene Post-Hoc Forschungsansätze")

    col_ph1, col_ph2 = st.columns(2)
    with col_ph1:
        st.markdown("#### 🔄 Kontrafaktische Erklärungen (Counterfactuals)")
        st.markdown(
            """
            - **Fragestellung:** *„Was ist die kleinste minimale Änderung an den Merkmalen, damit das Modell seine Entscheidung ändert?“*
            - **Beispiel Pinguin:** *„Wenn dieser Adelie-Pinguin 3 mm mehr Schnabellänge hätte, wäre er als Chinstrap klassifiziert worden.“*
            - **Praxisnutzen:** Hochrelevant für Endnutzer (z. B. Kreditvergabe oder Medizin: *„Was muss sich ändern, damit der Antrag genehmigt wird?“*).
            """
        )
    with col_ph2:
        st.markdown("#### 🎨 Saliency Maps & Grad-CAM (Computer Vision)")
        st.markdown(
            """
            - **Grad-CAM:** Berechnet Gradienten der Zielklasse bezüglich der letzten Faltungsschichten.
            - **Visualisierung:** Eine Heatmap zeigt pixelgenau, welche Bildbereiche für die Entscheidung ausschlaggebend waren.
            - **LRP (Layer-wise Relevance Propagation):** Propagiert Relevanzwerte deterministisch rückwärts durch alle Netzwerkschichten.
            """
        )

    st.markdown("---")
    st.markdown("### 3 · Gegenüberstellung: By Design vs. Post-Hoc")

    vergleich_data = {
        "Kriterium": [
            "Treue (Fidelity)",
            "Modellflexibilität",
            "Performance / Kapazität",
            "Erklärungsaufwand",
            "Typische Einsatzgebiete",
        ],
        "By Design (Intrinsisch)": [
            "100 % exakt (Modell IST die Erklärung)",
            "Eingeschränkt auf transparente Architekturen",
            "Exzellent bei Tabellendaten, schwieriger bei rohen Bildern/Audio",
            "Kein zusätzlicher Rechenaufwand nach dem Training",
            "Medizinische Diagnostik, Justiz, Kreditentscheidungen",
        ],
        "Post-Hoc (Nachträglich)": [
            "Approximation / Näherung (Gefahr von Scheinerklärungen)",
            "Universell modellagnostisch (jedes Modell nutzbar)",
            "Höchste Kapazität (Deep Learning, LLMs, Ensembles)",
            "Teilweise hoher zusätzlicher Rechenaufwand (Sampling)",
            "Computer Vision, NLP / LLMs, komplexe Ensembles",
        ],
    }
    st.dataframe(pd.DataFrame(vergleich_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 4 · Globale Erklärungsverfahren")
    st.markdown(
        """
        Während LIME und SHAP lokale Einzelfälle erklären, analysieren globale Verfahren das Gesamtverhalten des Modells:
        
        - **Permutation Feature Importance:** Misst den Genauigkeitsverlust des Modells, wenn ein Merkmal zufällig permutiert (zerstört) wird.
        - **Partial Dependence Plots (PDP):** Visualisieren den mittleren Verlauf der Modellvorhersage in Abhängigkeit eines Merkmals über den gesamten Datensatz.
        - **Accumulated Local Effects (ALE):** Berechnen lokale Vorhersageänderungen und sind insbesondere bei korrelierten Merkmalen robuster als PDPs.
        """
    )
    st.page_link("views/ml/explainable_ml.py", label="Mehr zu globalen xAI-Methoden in dieser App", icon="🗂️")

    st.markdown("---")
    st.markdown("### 5 · Explainable AI für neuronale Netze")
    st.markdown(
        """
        Für tiefe neuronale Netze nutzen spezialisierte Verfahren die interne Differenzierbarkeit und Schichtenstruktur:
        - **Saliency Maps:** Visualisieren Gradienten bezüglich der Eingabepixel.
        - **Integrated Gradients:** Integrieren Gradienten entlang eines Pfades von einer neutralen Baseline (z. B. schwarzes Bild) zur tatsächlichen Eingabe.
        - **DeepLIFT:** Propagiert Differenzen von Aktivierungen gegenüber einer Referenz rückwärts durch das Netzwerk.
        """
    )

    st.markdown("---")
    st.markdown("### 6 · Zentrale Herausforderungen der aktuellen XAI-Forschung")
    merkkasten(
        "Offene Forschungsfragen der Explainable AI",
        "• <b>1. Faithfulness vs. Plausibility:</b> Eine Erklärung kann für Menschen überzeugend aussehen, aber das tatsächliche (möglicherweise fehlerhafte) Modellverhalten verschleiern.<br>"
        "• <b>2. Adversarial Robustness:</b> LIME- und SHAP-Erklärungen können durch gezielte Störungen manipuliert werden, um diskriminierende Modelle nach außen 'fair' wirken zu lassen.<br>"
        "• <b>3. Kausalität statt Korrelation:</b> Reine Merkmalsattributionen zeigen Korrelationen, erfassen aber keine realen Kausalmechanismen.<br>"
        "• <b>4. Interpretierbarkeit von Large Language Models (LLMs):</b> Analyse und Steuerung von Milliarden Parametern in generativen Modellen (Mechanistic Interpretability, Chain-of-Thought).",
        typ="achtung",
    )

    st.markdown("---")
