"""Werkzeuge: Python Setup mit uv und Visual Studio Code.

Führt von null bis zur laufenden Streamlit-App: uv installieren, Python und
Umgebung anlegen, Pakete verwalten, VS Code einrichten, typische Fehler.
"""

import streamlit as st

from utils.theming import kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "🐍",
    "Python Setup",
    "Mit uv und Visual Studio Code in einer halben Stunde arbeitsfähig",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Für die Gruppenprojekte brauchst du Python auf deinem eigenen Rechner. Der
klassische Weg dorthin ist bekannt dafür, dass er schiefgeht: Man installiert
Python von der Webseite, dann noch eines über Anaconda, irgendwann gibt es
drei Versionen, `pip install` landet in der falschen davon, und ein Paket, das
gestern lief, tut es heute nicht mehr.

Wir nehmen deshalb einen kürzeren Weg. Zwei Werkzeuge reichen:

- **uv** kümmert sich um Python selbst, um die Projektumgebung und um alle
  Pakete.
- **Visual Studio Code** ist der Editor, in dem du schreibst, ausführst und
  Fehler suchst.

Beides ist kostenlos, läuft auf Windows, macOS und Linux und ist in wenigen
Minuten eingerichtet.
"""
)

# ------------------------------------------------------------ Warum uv?
st.markdown("## Warum uv?")
st.markdown(
    """
`uv` ist ein junges Werkzeug (von den Entwicklern des Linters `ruff`), das
mehrere alte Werkzeuge in einem bündelt. Statt `pyenv` für Python-Versionen,
`venv` für Umgebungen und `pip` für Pakete gibt es nur noch `uv`. Es ist
außerdem sehr schnell, Installationen dauern Sekunden statt Minuten.

Wichtiger als die Geschwindigkeit ist für uns die **Reproduzierbarkeit**. `uv`
schreibt in eine Datei `pyproject.toml`, welche Pakete dein Projekt braucht,
und in `uv.lock`, in welcher Version sie tatsächlich installiert wurden. Wer
euer Projekt später klont, bekommt mit einem einzigen Befehl exakt dieselbe
Umgebung. Für eine Gruppenarbeit, die von mehreren Rechnern aus bearbeitet
wird, ist das genau der Unterschied zwischen „läuft“ und „läuft bei mir“.
"""
)

merkkasten(
    "Definition",
    "Eine <b>virtuelle Umgebung</b> (Ordner <code>.venv</code>) ist eine "
    "abgeschottete Python-Installation für genau ein Projekt. Pakete, die du "
    "dort installierst, sehen andere Projekte nicht. Deshalb kann Projekt A "
    "mit einer alten und Projekt B mit einer neuen Version desselben Pakets "
    "arbeiten, ohne dass sich beide in die Quere kommen. Der Ordner "
    "<code>.venv</code> gehört nie ins Git-Repository, er lässt sich jederzeit "
    "neu erzeugen.",
    typ="definition",
)

# --------------------------------------------------- Schritt 1: uv installieren
st.markdown("## Schritt 1: uv installieren")
st.markdown(
    "Öffne ein Terminal (unter Windows die **PowerShell**, unter macOS das "
    "Programm **Terminal**) und führe die Zeile für dein System aus."
)

tab_windows, tab_mac = st.tabs(["Windows", "macOS und Linux"])

with tab_windows:
    st.code(
        'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        language="powershell",
    )
    st.markdown(
        "Alternativ über den Paketmanager: `winget install --id=astral-sh.uv -e`."
    )

with tab_mac:
    st.code("curl -LsSf https://astral.sh/uv/install.sh | sh", language="bash")
    st.markdown("Alternativ mit Homebrew: `brew install uv`.")

st.markdown(
    """
Danach schließt du das Terminal einmal und öffnest es neu, damit der neue
Befehl gefunden wird. Zur Kontrolle:
"""
)
st.code("uv --version", language="bash")

# ------------------------------------------------- Schritt 2: Python holen
st.markdown("## Schritt 2: Python besorgen")
st.markdown(
    """
Du musst Python **nicht** von Hand installieren. `uv` lädt die passende
Version selbst herunter und legt sie an einen Ort, an dem sie nichts kaputt
macht.
"""
)
st.code(
    """uv python install 3.12     # eine Version installieren
uv python list             # zeigt, was verfügbar und installiert ist""",
    language="bash",
)
st.markdown(
    "Version 3.12 ist eine gute Wahl: neu genug für alle Pakete, die wir "
    "brauchen, und alt genug, dass es keine Überraschungen gibt."
)

# ------------------------------------------- Schritt 3: eigenes Projekt
st.markdown("## Schritt 3: Ein eigenes Projekt anlegen")
st.markdown(
    """
Für ein neues Gruppenprojekt legst du einen Ordner an und lässt `uv` das
Gerüst erzeugen. Der Befehl `uv add` installiert ein Paket und schreibt es
zugleich in die Projektdatei, du musst also nichts von Hand pflegen.
"""
)
st.code(
    """uv init gletscher-gang          # Ordner und pyproject.toml anlegen
cd gletscher-gang

uv add pandas scikit-learn streamlit plotly
uv add --dev pytest             # nur zum Entwickeln, nicht zum Ausführen

uv run python analyse.py        # Skript in der Projektumgebung starten
uv run streamlit run app.py     # Streamlit-App starten""",
    language="bash",
)
st.markdown(
    """
Der entscheidende Befehl ist **`uv run`**. Er sorgt dafür, dass dein Code in
der Projektumgebung läuft, und legt diese Umgebung beim ersten Mal automatisch
an. Du musst keine Umgebung „aktivieren“ und kannst dich auch nicht in der
falschen befinden.
"""
)

st.markdown("### Diese Website lokal starten")
st.markdown(
    """
Das Repository hier benutzt noch die klassische `requirements.txt`. Auch damit
kommt `uv` zurecht:
"""
)
st.code(
    """git clone https://github.com/JanTeichertKluge/causality-leysin.git
cd causality-leysin

uv venv                                  # .venv anlegen
uv pip install -r requirements.txt       # Pakete hineininstallieren
uv run streamlit run streamlit_app.py    # App starten""",
    language="bash",
)
st.markdown(
    "Der letzte Befehl öffnet den Browser auf `http://localhost:8501`. "
    "Beenden im Terminal mit `Strg + C`."
)

st.markdown("### Die Befehle im Vergleich")
st.markdown(
    """
| Aufgabe | klassisch | mit uv |
|---|---|---|
| Umgebung anlegen | `python -m venv .venv` | `uv venv` |
| Umgebung aktivieren | `.venv\\Scripts\\activate` | nicht nötig, `uv run` reicht |
| Paket installieren | `pip install pandas` | `uv add pandas` |
| Paket entfernen | `pip uninstall pandas` | `uv remove pandas` |
| Alles nachinstallieren | `pip install -r requirements.txt` | `uv sync` |
| Skript starten | `python analyse.py` | `uv run python analyse.py` |
| Python-Version wechseln | separates Werkzeug nötig | `uv python install 3.12` |
"""
)

# ------------------------------------------------- Schritt 4: VS Code
st.markdown("## Schritt 4: Visual Studio Code einrichten")
st.markdown(
    """
[Visual Studio Code](https://code.visualstudio.com) ist der Editor, den wir in
der Akademie benutzen. Nach der Installation sind vier Handgriffe nötig.

**1. Erweiterungen installieren.** Links in der Leiste auf das Symbol für
Extensions (die vier Kästchen), dann nach **Python** von Microsoft suchen und
installieren. Damit kommen Autovervollständigung, Fehlerhinweise und der
Debugger. Wer mit Notebooks arbeiten will, installiert zusätzlich **Jupyter**.

**2. Den Projektordner öffnen.** Menü `File` und dann `Open Folder`, und zwar
den Projektordner selbst, nicht eine einzelne Datei. VS Code richtet sich
immer am geöffneten Ordner aus: Dort sucht es die Umgebung, dort startet das
Terminal.

**3. Den Interpreter auswählen.** Drücke `Strg + Shift + P` (auf dem Mac
`Cmd + Shift + P`), tippe `Python: Select Interpreter` und wähle den Eintrag,
der auf `.venv` zeigt. Unten rechts in der Statusleiste steht danach die
gewählte Version. Dieser Schritt wird am häufigsten vergessen, und fast jedes
„das Paket ist doch installiert“ geht darauf zurück.

**4. Das eingebaute Terminal benutzen.** Menü `Terminal` und dann
`New Terminal` öffnet eine Kommandozeile direkt im Projektordner. Dort tippst
du deine `uv`-Befehle. Ein Skript startest du entweder mit `uv run python
datei.py` oder mit dem Abspielknopf oben rechts.
"""
)

with vertiefung("Notebooks, Debugger und Streamlit in VS Code"):
    st.markdown(
        """
    **Zellen ohne Notebook.** Schreibst du in eine normale `.py`-Datei die
    Zeile `# %%`, behandelt VS Code den Abschnitt darunter als Zelle und bietet
    „Run Cell“ an. Das Ergebnis erscheint im interaktiven Fenster. Du bekommst
    also den Komfort eines Notebooks, behältst aber eine saubere Python-Datei,
    die sich mit Git vernünftig vergleichen lässt.

    **Debugger statt print.** Ein Klick links neben eine Zeilennummer setzt
    einen roten Punkt, den Breakpoint. Mit `F5` startest du das Skript, es
    hält an dieser Stelle an, und du kannst alle Variablen ansehen. Das ist
    fast immer schneller, als Ausgaben in den Code zu streuen.

    **Streamlit im Debugger.** Weil Streamlit als Modul startet, braucht es
    eine kleine Konfiguration. Lege eine Datei `.vscode/launch.json` an:

    ```json
    {
      "version": "0.2.0",
      "configurations": [
        {
          "name": "Streamlit",
          "type": "debugpy",
          "request": "launch",
          "module": "streamlit",
          "args": ["run", "streamlit_app.py"]
        }
      ]
    }
    ```

    Danach kannst du die App mit `F5` starten und in deinem Seitencode
    Breakpoints setzen.

    **Was in pyproject.toml und uv.lock steht.** In `pyproject.toml` stehen
    Name, Python-Version und die Pakete, die du bewusst angefordert hast. In
    `uv.lock` steht zusätzlich jedes indirekt mitinstallierte Paket mit exakter
    Version und Prüfsumme. Beide Dateien gehören ins Repository, `.venv`
    dagegen nicht.
    """
    )

# ---------------------------------------------------------- Fehlersuche
st.markdown("## Wenn etwas nicht funktioniert")
st.markdown(
    """
Vier Fehler treten so oft auf, dass sich Nachschlagen lohnt.

**`uv: command not found` oder `uv wird nicht erkannt`.** Das Terminal kennt
den neuen Befehl noch nicht. Terminal schließen, neu öffnen, notfalls den
Rechner neu starten.

**`ModuleNotFoundError`, obwohl das Paket installiert ist.** Dein Code läuft in
einer anderen Umgebung als der, in die du installiert hast. Entweder du
startest mit `uv run`, oder du wählst in VS Code den Interpreter aus `.venv`
(Schritt 4, Punkt 3).

**`Port 8501 is already in use`.** Es läuft noch eine Streamlit-App. Entweder
das alte Terminal mit `Strg + C` beenden oder einen anderen Port nehmen:
`uv run streamlit run streamlit_app.py --server.port 8502`.

**PowerShell verweigert das Aktivierungsskript.** Windows blockiert Skripte
standardmäßig. Der einfachste Ausweg ist, gar nicht zu aktivieren, sondern
alles über `uv run` zu starten. Wer es dennoch braucht:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
"""
)

merkkasten(
    "Die drei Befehle, die reichen",
    "<code>uv add paket</code> zum Installieren, <code>uv run …</code> zum "
    "Starten und <code>uv sync</code>, um auf einem neuen Rechner alles "
    "nachzuziehen. Alles Weitere kannst du nachschlagen, wenn du es brauchst.",
    typ="merke",
)

st.markdown("## Weiterführende Links")
st.markdown(
    """
- [uv-Dokumentation](https://docs.astral.sh/uv/) mit Einstieg und vollständiger Befehlsreferenz
- [Python in VS Code](https://code.visualstudio.com/docs/languages/python) als offizielles Tutorial
- [Der Python-Tutorial-Klassiker](https://docs.python.org/3/tutorial/) für die Sprache selbst
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    """
Wenn `uv run streamlit run streamlit_app.py` bei dir eine Website öffnet, ist
alles bereit. Dann kannst du in die Kapitel einsteigen oder direkt mit dem
Gerüst für euer Gruppenprojekt anfangen.
"""
)

weiter_streamlit, weiter_ml = st.columns(2)
with weiter_streamlit:
    st.page_link(
        "views/streamlit_intro.py", label="Zurück: Was ist Streamlit?", icon="🎈"
    )
with weiter_ml:
    st.page_link(
        "views/ml/grundlagen.py", label="Weiter: Was ist Maschinelles Lernen?", icon="🤖"
    )
