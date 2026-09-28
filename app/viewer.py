"""Read-only dashboard and replay for the accepted debate experiment.

Run from the repository root:

    .venv/bin/streamlit run app/viewer.py

The app reads verified exports and SQLite through ``mad.viewer_data``. It makes
no model calls, never writes to SQLite and never opens a benchmark key file.
"""

from __future__ import annotations

import html
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import altair as alt
import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mad.viewer_data import (  # noqa: E402
    AgentResponseView,
    ExperimentOverview,
    QuestionReplay,
    ViewerDataError,
    load_experiment_overview,
    load_question_replay,
)


DEFAULT_QUESTION_ID = "mmlu_pro_v1:test:5503"

CURATED_QUESTIONS = {
    "Wrong majority corrected": "mmlu_pro_v1:test:5503",
    "Deadlock resolved correctly": "mmlu_pro_v1:test:11081",
    "Correct majority lost": "mmlu_pro_v1:test:5104",
    "Unanimous but wrong": "mmlu_pro_v1:test:1203",
    "Disagreement never resolved": "mmlu_pro_v1:test:2697",
    "Three failures, then recovery": "mmlu_pro_v1:test:11994",
}

# The one-line description shown under the chosen case.
CURATED_NOTES = {
    "mmlu_pro_v1:test:5503": "Round 1 settled on C. After reading peer reasoning all five moved to A, the benchmark answer.",
    "mmlu_pro_v1:test:11081": "No Round 1 majority. Round 2 reached a unanimous, correct answer.",
    "mmlu_pro_v1:test:5104": "Round 1 held a correct majority. Round 2 lost it and reached no consensus at all.",
    "mmlu_pro_v1:test:1203": "No Round 1 majority. Round 2 agreed unanimously on D, which is not the benchmark answer.",
    "mmlu_pro_v1:test:2697": "The group disagreed in both rounds, so no answer was recorded either time.",
    "mmlu_pro_v1:test:11994": "Three agents failed in Round 1, leaving too few votes to reach three. Round 2 completed and was correct.",
}

STEPS = (
    ("question", "The question"),
    ("round1_request", "Round 1 request"),
    ("round1_responses", "Round 1 responses"),
    ("parsing", "Parsing the answers"),
    ("round1_vote", "First group vote"),
    ("round2_input", "What each agent receives in Round 2"),
    ("round2_responses", "Round 2 responses"),
    ("round2_vote", "Final group vote"),
    ("summary", "Question summary"),
)

MODEL_LABELS = {
    "agent_llama": "Llama",
    "agent_qwen": "Qwen",
    "agent_mistral": "Mistral",
    "agent_deepseek": "DeepSeek",
    "agent_gemma": "Gemma",
}

BLUE = "#1F4E79"
INK_2 = "#45423C"
MUTED = "#6E6B65"
RULE = "#DEDBD4"
SANS = "IBM Plex Sans, sans-serif"
VIOLET = "#4E7BA6"
GREEN = "#2D6A4F"
AMBER = "#D97706"
RED = "#DC2626"
SLATE = "#64748B"


@st.cache_data(show_spinner=False)
def _load_overview() -> ExperimentOverview:
    return load_experiment_overview()


@st.cache_data(show_spinner=False)
def _load_replay(question_id: str) -> QuestionReplay:
    return load_question_replay(question_id)


def _inject_styles() -> None:
    """One stylesheet, built from a small token set.

    The look is a laboratory instrument rather than a dashboard: warm paper
    ground, one archival blue, and every stored datum set in mono so an ID, a
    letter and a token count all read as measurements. Colour is reserved for
    meaning - correct, incorrect, failed - and never used as decoration.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --paper:   #F6F5F2;
            --surface: #FFFFFF;
            --sunk:    #F0EEE9;
            --ink:     #1C1B19;
            --ink-2:   #45423C;
            --muted:   #6E6B65;
            --rule:    #DEDBD4;
            --rule-2:  #C6C2B8;
            --accent:  #1F4E79;
            --accent-w:#E7EEF5;
            --correct: #2D6A4F;
            --wrong:   #9B2226;
            --flag:    #9A6212;

            --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;

            --r-sm: 3px;
            --r-md: 6px;
        }

        html, body, .stApp { font-family: var(--sans); }
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: var(--sans); }
        /* Streamlit draws icons as ligatures; overriding their font prints the
           ligature name as literal text. Leave them alone. */
        [data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
        span[class*="material-symbols"], span[class*="material-icons"] {
            font-family: "Material Symbols Rounded", "Material Icons" !important;
        }
        .stApp { background: var(--paper); }
        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stAppViewContainer"] { overflow-x: hidden; }
        .block-container {
            max-width: 1360px;
            padding-top: 2.4rem;
            padding-bottom: 5rem;
        }

        h1, h2, h3, h4 { color: var(--ink); font-weight: 600; letter-spacing: -0.01em; }
        p, li, label, [data-testid="stCaptionContainer"] { color: var(--ink-2); }
        [data-testid="stCaptionContainer"] p { color: var(--muted); font-size: .78rem; }

        /* Data reads as data. */
        .mono, .metric-value, .option-letter, .vote-letter,
        .eyebrow, .metric-label, .section-kicker, .badge, .pipeline-number {
            font-family: var(--mono);
            font-variant-numeric: tabular-nums;
        }

        /* ---- sidebar ---- */
        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--rule);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.6rem; }

        .brand { margin: 0 0 1.5rem; }
        .brand-title {
            font-family: var(--mono);
            font-size: .95rem;
            font-weight: 600;
            color: var(--ink);
            letter-spacing: -0.01em;
        }
        .brand-subtitle {
            font-size: .76rem;
            color: var(--muted);
            margin-top: .3rem;
            line-height: 1.45;
        }
        .brand-rule { height: 2px; width: 2.2rem; background: var(--accent); margin: .7rem 0 0; }

        /* ---- page header ---- */
        .eyebrow {
            color: var(--muted);
            font-size: .7rem;
            font-weight: 500;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin-bottom: .6rem;
        }
        .page-title {
            color: var(--ink);
            font-size: 1.95rem;
            line-height: 1.18;
            font-weight: 600;
            letter-spacing: -0.025em;
            margin: 0;
            text-wrap: balance;
        }
        .page-copy {
            max-width: 68ch;
            color: var(--ink-2);
            font-size: .95rem;
            line-height: 1.62;
            margin: .75rem 0 1.7rem;
        }

        /* ---- Streamlit columns must stretch, or cards in a row go ragged ---- */
        [data-testid="stHorizontalBlock"] { align-items: stretch; }
        [data-testid="stColumn"] > div,
        [data-testid="stColumn"] [data-testid="stVerticalBlock"] { height: 100%; }

        /* ---- metric cards ---- */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(var(--cards, 5), minmax(0, 1fr));
            gap: .7rem;
            margin-bottom: 1.4rem;
        }
        .metric-card {
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: .35rem;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-top: 2px solid var(--rule-2);
            border-radius: var(--r-md);
            padding: .95rem 1rem 1rem;
        }
        .metric-accent-blue    { border-top-color: var(--accent); }
        .metric-accent-violet  { border-top-color: #4E7BA6; }
        .metric-accent-green   { border-top-color: var(--correct); }
        .metric-accent-amber   { border-top-color: var(--flag); }
        .metric-label {
            color: var(--muted);
            font-size: .68rem;
            font-weight: 500;
            letter-spacing: .07em;
            text-transform: uppercase;
        }
        .metric-value {
            color: var(--ink);
            font-size: clamp(1.15rem, 1.5vw, 1.5rem);
            overflow-wrap: anywhere;
            line-height: 1.1;
            font-weight: 600;
            letter-spacing: -0.03em;
            margin-top: .1rem;
        }
        .metric-note {
            color: var(--muted);
            font-size: .77rem;
            line-height: 1.45;
            margin-top: auto;
            padding-top: .35rem;
        }

        /* ---- finding block (replaces the gradient card) ---- */
        .finding {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
            gap: .2rem 2.2rem;
            align-items: start;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-left: 3px solid var(--accent);
            border-radius: var(--r-sm);
            padding: 1.05rem 1.35rem 1.15rem;
        }
        .finding-kicker {
            grid-column: 1 / -1;
            font-family: var(--mono);
            color: var(--accent);
            font-size: .68rem;
            font-weight: 500;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .finding-title {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 600;
            line-height: 1.38;
            margin: .45rem 0 0;
            text-wrap: balance;
        }
        .finding-copy {
            color: var(--ink-2);
            line-height: 1.62;
            font-size: .88rem;
            margin-top: .5rem;
            max-width: 62ch;
        }

        /* ---- panel heading ---- */
        .section-kicker {
            color: var(--muted);
            font-size: .66rem;
            font-weight: 500;
            letter-spacing: .09em;
            text-transform: uppercase;
            margin-bottom: .3rem;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 600;
            line-height: 1.35;
            margin-bottom: .2rem;
        }
        .section-copy {
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.5;
            margin-bottom: .9rem;
            max-width: 62ch;
        }

        /* ---- integrity strip ---- */
        .integrity-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
        }
        .integrity-item {
            background: var(--sunk);
            border: 1px solid var(--rule);
            border-radius: var(--r-sm);
            padding: .8rem .9rem;
        }
        .integrity-item strong {
            display: block;
            color: var(--ink);
            font-size: .82rem;
            font-weight: 600;
            margin-bottom: .22rem;
        }
        .integrity-item span {
            display: block;
            font-family: var(--mono);
            color: var(--muted);
            font-size: .73rem;
            line-height: 1.5;
        }

        /* ---- pipeline: a real flex strip, arrows are items not overlays ---- */
        .pipeline-scroll { margin: .2rem 0 1.4rem; max-width: 100%; }
        .pipeline {
            display: flex;
            align-items: stretch;
            gap: 0;
            width: 100%;
        }
        .pipeline-node {
            display: flex;
            flex: 1 1 0;
            min-width: 0;
            flex-direction: column;
            gap: .18rem;
            justify-content: flex-start;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--r-sm);
            padding: .6rem .75rem .65rem;
        }
        .pipeline-node.is-active {
            border-color: var(--accent);
            background: var(--accent-w);
        }
        .pipeline-arrow {
            display: flex;
            flex: 0 0 auto;
            align-items: center;
            padding: 0 .32rem;
            color: var(--rule-2);
            font-size: .85rem;
        }
        .pipeline-number { color: var(--muted); font-size: .64rem; font-weight: 500; letter-spacing: .06em; }
        .pipeline-node.is-active .pipeline-number { color: var(--accent); }
        .pipeline-label { color: var(--ink); font-size: .78rem; font-weight: 600; line-height: 1.28; overflow-wrap: anywhere; }
        .pipeline-detail { color: var(--muted); font-size: .68rem; line-height: 1.3; overflow-wrap: anywhere; }

        /* ---- question and options ---- */
        .question-card {
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--r-md);
            padding: 1.15rem 1.25rem;
            margin-bottom: .85rem;
        }
        .option-card {
            display: flex;
            align-items: baseline;
            gap: .65rem;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--r-sm);
            padding: .6rem .8rem;
            margin-bottom: .4rem;
            color: var(--ink-2);
            font-size: .88rem;
            line-height: 1.5;
        }
        .option-letter {
            flex: 0 0 auto;
            color: var(--accent);
            background: var(--accent-w);
            border-radius: var(--r-sm);
            padding: .08rem .42rem;
            font-size: .8rem;
            font-weight: 600;
        }

        /* ---- vote cards: one baseline across all five ---- */
        .vote-card {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: .45rem;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--r-md);
            padding: .8rem .6rem .85rem;
            text-align: center;
        }
        .vote-agent {
            font-size: .74rem;
            font-weight: 600;
            color: var(--muted);
            letter-spacing: .02em;
            min-height: 1.15rem;
        }
        .vote-letter {
            font-size: 2.1rem;
            font-weight: 600;
            line-height: 1;
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        .vote-letter.is-missing { color: var(--rule-2); }

        /* ---- status chips ---- */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: .3rem;
            border-radius: 999px;
            padding: .16rem .55rem;
            font-size: .68rem;
            font-weight: 500;
            letter-spacing: .03em;
            border: 1px solid transparent;
        }
        .badge-blue  { color: var(--accent);  background: var(--accent-w); border-color: #CBDCEA; }
        .badge-green { color: var(--correct); background: #E4EFE8;         border-color: #C6DCCF; }
        .badge-amber { color: var(--flag);    background: #F6EEDC;         border-color: #E6D6B4; }
        .badge-slate { color: var(--muted);   background: var(--sunk);     border-color: var(--rule); }

        /* ---- Streamlit primitives ---- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--rule) !important;
            border-radius: var(--r-md) !important;
            box-shadow: none !important;
        }
        [data-testid="stMetric"] { background: transparent; padding: .2rem 0; }
        /* Streamlit sets a 2rem value and clips the overflow with an ellipsis,
           which silently truncated "DigitalOcean" and the cost. Smaller, and
           allowed to wrap instead of disappearing. */
        [data-testid="stMetricValue"] {
            font-family: var(--mono);
            color: var(--ink);
            font-size: 1.12rem;
            line-height: 1.3;
            letter-spacing: -.01em;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricValue"] > div { overflow: visible; text-overflow: clip; }
        [data-testid="stMetricLabel"] p { font-size: .72rem; color: var(--muted); }
        [data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: var(--r-sm); overflow: hidden; }
        [data-testid="stExpander"] details {
            border: 1px solid var(--rule) !important;
            border-radius: var(--r-sm) !important;
            background: var(--surface);
        }
        code, pre, [data-testid="stCode"] { font-family: var(--mono) !important; font-size: .8rem !important; }

        [data-baseweb="tab-list"] { gap: .1rem; border-bottom: 1px solid var(--rule); }
        [data-baseweb="tab"] {
            border-radius: 0;
            padding: .5rem .9rem;
            background: transparent;
            font-size: .85rem;
            color: var(--muted);
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: transparent;
            color: var(--ink);
            box-shadow: inset 0 -2px 0 var(--accent);
        }

        .stButton > button {
            border-radius: var(--r-sm);
            min-height: 2.5rem;
            border: 1px solid var(--rule-2);
            background: var(--surface);
            color: var(--ink-2);
            font-weight: 500;
            font-size: .85rem;
        }
        .stButton > button:hover:not(:disabled) {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--surface);
        }
        .stButton > button:disabled { opacity: .45; }
        hr { border-color: var(--rule); }

        @media (max-width: 900px) {
            .integrity-strip { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .finding { grid-template-columns: 1fr; }
            .page-title { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _brand() -> None:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">Multi-agent debate</div>
            <div class="brand-subtitle">Replay of the accepted 300-question experiment</div>
            <div class="brand-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_header(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow">{html.escape(eyebrow)}</div>
        <h1 class="page-title">{html.escape(title)}</h1>
        <p class="page-copy">{html.escape(copy)}</p>
        """,
        unsafe_allow_html=True,
    )


def _metric_row(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """Render metric cards as one CSS grid so every card shares a height.

    Streamlit columns do not equalise the height of their children, so five
    separate cards went ragged as soon as one note wrapped onto a second line.
    Grid rows stretch to the tallest cell by definition.
    """
    cells = "".join(
        f'<div class="metric-card metric-accent-{accent}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</div>"
        for label, value, note, accent in cards
    )
    st.markdown(
        f'<div class="metric-grid" style="--cards:{len(cards)}">{cells}</div>',
        unsafe_allow_html=True,
    )


def _panel_heading(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def _base_chart(chart: alt.Chart) -> alt.Chart:
    """One chart theme, matching the page palette.

    The legend sits bottom-left and wraps into columns. Centred bottom legends
    were being clipped at the container edge, which silently truncated series
    names like "Became incorrect" to "Be".
    """
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=RULE,
            domain=False,
            tickColor=RULE,
            labelFontSize=11,
            titleFontSize=11,
            labelFont=SANS,
            titleFont=SANS,
        )
        .configure_legend(
            labelColor=INK_2,
            titleColor=MUTED,
            symbolType="circle",
            orient="bottom",
            direction="horizontal",
            columns=2,
            labelLimit=0,
            labelFontSize=11,
            titleFontSize=11,
            labelFont=SANS,
            titleFont=SANS,
            offset=14,
            padding=0,
            title=None,
        )
        .configure_axisX(labelAngle=0)
    )


def _render_accuracy_chart(overview: ExperimentOverview) -> None:
    rows = [
        {
            "Round": f"Round {result.round}",
            "Accuracy": result.accuracy_percent,
            "Correct": result.correct,
        }
        for result in overview.group_accuracy
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=70)
        .encode(
            x=alt.X("Round:N", title=None, sort=["Round 1", "Round 2"]),
            y=alt.Y("Accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Round:N",
                scale=alt.Scale(domain=["Round 1", "Round 2"], range=[BLUE, GREEN]),
                legend=None,
            ),
            tooltip=["Round:N", alt.Tooltip("Accuracy:Q", format=".1f"), "Correct:Q"],
        )
    )
    labels = chart.mark_text(dy=-12, color="#111827", fontWeight=800, fontSize=14).encode(
        text=alt.Text("Accuracy:Q", format=".1f")
    )
    st.altair_chart(_base_chart((chart + labels).properties(height=300)), width="stretch")


def _render_transition_chart(overview: ExperimentOverview) -> None:
    rows = [
        {"Outcome": "Stayed correct", "Questions": overview.stayed_correct},
        {"Outcome": "Became correct", "Questions": overview.became_correct},
        {"Outcome": "Became incorrect", "Questions": overview.became_incorrect},
        {"Outcome": "Stayed incorrect", "Questions": overview.stayed_incorrect},
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_arc(innerRadius=62, outerRadius=104, cornerRadius=5, padAngle=0.02)
        .encode(
            theta=alt.Theta("Questions:Q"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=[
                        "Stayed correct",
                        "Became correct",
                        "Became incorrect",
                        "Stayed incorrect",
                    ],
                    range=[BLUE, GREEN, RED, "#CBD5E1"],
                ),
                title=None,
            ),
            tooltip=["Outcome:N", "Questions:Q"],
        )
        .properties(height=290)
    )
    st.altair_chart(_base_chart(chart), width="stretch")


def _render_agent_chart(overview: ExperimentOverview) -> None:
    rows = [
        {
            "Model": MODEL_LABELS[result.agent_id],
            "Round": f"Round {result.round}",
            "Accuracy": result.accuracy_percent,
            "Failures": result.failure_count,
        }
        for result in overview.agent_accuracy
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", title=None, sort=list(MODEL_LABELS.values())),
            xOffset=alt.XOffset("Round:N"),
            y=alt.Y("Accuracy:Q", title="Accuracy on valid answers (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Round:N",
                scale=alt.Scale(domain=["Round 1", "Round 2"], range=[BLUE, VIOLET]),
                title=None,
            ),
            tooltip=["Model:N", "Round:N", alt.Tooltip("Accuracy:Q", format=".1f"), "Failures:Q"],
        )
        .properties(height=300)
    )
    st.altair_chart(_base_chart(chart), width="stretch")


def _render_consensus_chart(overview: ExperimentOverview) -> None:
    state_order = ["UNANIMOUS", "CONSENSUS", "NO_CONSENSUS", "INSUFFICIENT_ANSWERS"]
    rows = [
        {
            "Round": f"Round {result.round}",
            "State": result.state.replace("_", " ").title(),
            "RawState": result.state,
            "Questions": result.count,
        }
        for result in overview.consensus_states
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadius=5, size=58)
        .encode(
            y=alt.Y("Round:N", title=None, sort=["Round 1", "Round 2"]),
            x=alt.X("Questions:Q", title="Questions"),
            color=alt.Color(
                "State:N",
                scale=alt.Scale(
                    domain=[state.replace("_", " ").title() for state in state_order],
                    range=[GREEN, BLUE, AMBER, RED],
                ),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("RawState:N", sort="ascending"),
            tooltip=["Round:N", "State:N", "Questions:Q"],
        )
        .properties(height=286)
    )
    st.altair_chart(_base_chart(chart), width="stretch")


def _render_overview(overview: ExperimentOverview) -> None:
    _page_header(
        "Accepted experiment · 300 questions",
        "What changed after the models debated?",
        "A verified summary of the final experiment. Five different model families answered independently, then reconsidered after reading anonymous peer reasoning.",
    )

    round1, round2 = sorted(overview.group_accuracy, key=lambda item: item.round)
    _metric_row((
        ("Round 1 accuracy", f"{round1.accuracy_percent:.1f}%",
         f"{round1.correct} of {round1.questions} correct", "blue"),
        ("Round 2 accuracy", f"{round2.accuracy_percent:.1f}%",
         f"{round2.correct} of {round2.questions} correct", "violet"),
        ("Debate effect", f"+{overview.effect_points:.2f} pts",
         f"95% CI {overview.confidence_low_points:+.2f} to {overview.confidence_high_points:+.2f}", "green"),
        ("Experiment cost", f"${overview.total_cost_usd:.2f}",
         f"{overview.total_tokens / 1_000_000:.2f}M stored tokens", "amber"),
        ("Runtime", f"{overview.wall_clock_hours:.1f} h",
         f"{overview.total_failures} failures recorded", "blue"),
    ))

    # The finding runs full width and the two charts pair off beneath it.
    # Stacking the finding above one chart made the right column far taller
    # than the left, leaving a long empty gap under the accuracy card.
    st.markdown(
        f"""
        <div class="finding">
            <div class="finding-kicker">Central finding</div>
            <div class="finding-title">Debate helped, mainly when the group was initially undecided.</div>
            <div class="finding-copy">
                {overview.corrected_from_undecided} of {overview.became_correct} corrections began without a Round 1
                majority. Debate was more useful for resolving disagreement than for repairing an
                already-decided answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            _panel_heading("Headline result", "Group accuracy by round", "The same 300 questions were scored before and after communication.")
            _render_accuracy_chart(overview)
    with right:
        with st.container(border=True):
            _panel_heading("Question movement", "What changed between rounds?", "Every question appears once in the four transition groups.")
            _render_transition_chart(overview)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            _panel_heading("Model behaviour", "Individual accuracy", "Accuracy is calculated over each agent's valid parsed answers; failures are reported separately.")
            _render_agent_chart(overview)
    with right:
        with st.container(border=True):
            _panel_heading("Collective behaviour", "Consensus changed", "A group answer always required three matching votes out of five.")
            _render_consensus_chart(overview)

    st.write("")
    with st.container(border=True):
        _panel_heading("Research integrity", "What this dashboard guarantees", "Presentation is separated from execution and scoring.")
        st.markdown(
            f"""
            <div class="integrity-strip">
                <div class="integrity-item"><strong>Read-only evidence</strong><span>No API calls and no database writes</span></div>
                <div class="integrity-item"><strong>Frozen experiment</strong><span>{html.escape(overview.settings_version)} · {overview.question_count} questions</span></div>
                <div class="integrity-item"><strong>Verified evaluation</strong><span>{html.escape(overview.evaluation_version)} · McNemar p={overview.mcnemar_p_value:.3f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _response_map(replay: QuestionReplay, round_number: int) -> dict[str, AgentResponseView]:
    round_replay = replay.round1 if round_number == 1 else replay.round2
    return {response.agent_id: response for response in round_replay.responses}


def _answer_change_rows(replay: QuestionReplay) -> list[dict[str, object]]:
    first = _response_map(replay, 1)
    second = _response_map(replay, 2)
    rows = []
    for agent_id, initial in first.items():
        revised = second[agent_id]
        changed = (
            initial.extracted_letter is not None
            and revised.extracted_letter is not None
            and initial.extracted_letter != revised.extracted_letter
        )
        rows.append(
            {
                "Agent": initial.display_name,
                "Round 1": initial.extracted_letter or "No vote",
                "Round 2": revised.extracted_letter or "No vote",
                "Changed": "Yes" if changed else "No",
                "Status": revised.status,
            }
        )
    return rows


def _total_usage(replay: QuestionReplay) -> dict[str, float | int]:
    outcomes = (replay.round1.outcome, replay.round2.outcome)
    return {
        "prompt_tokens": sum(outcome.total_prompt_tokens for outcome in outcomes),
        "completion_tokens": sum(outcome.total_completion_tokens for outcome in outcomes),
        "cost_usd": sum(outcome.total_cost_usd for outcome in outcomes),
        "latency_seconds": sum(outcome.total_latency_seconds for outcome in outcomes),
    }


def _render_navigation(step_index: int) -> None:
    previous, position, following = st.columns([1, 2, 1])
    with previous:
        if st.button("← Previous", disabled=step_index == 0, width="stretch"):
            st.session_state.replay_step = step_index - 1
            st.rerun()
    with position:
        st.markdown(
            f"<p style='text-align:center;margin:.55rem 0 0;color:#64748b;font-weight:650'>Step {step_index + 1} of {len(STEPS)}</p>",
            unsafe_allow_html=True,
        )
    with following:
        if st.button("Next →", disabled=step_index == len(STEPS) - 1, width="stretch"):
            st.session_state.replay_step = step_index + 1
            st.rerun()


def _render_pipeline(step_index: int) -> None:
    nodes = (
        ("01", "Question", "Frozen input"),
        ("02", "Independent answers", "Five model families"),
        ("03", "Round 1 vote", "Threshold stays at 3"),
        ("04", "Peer exchange", "Anonymous reasoning"),
        ("05", "Reconsideration", "New response"),
        ("06", "Final vote", "Stored outcome"),
    )
    active_map = (0, 1, 1, 2, 2, 3, 4, 5, 5)
    active = active_map[step_index]
    rendered = []
    for index, (number, label, detail) in enumerate(nodes):
        if index:
            # A real flex item, so it can never sit on top of the next border.
            rendered.append('<div class="pipeline-arrow">&rarr;</div>')
        state = " is-active" if index == active else ""
        rendered.append(
            f'<div class="pipeline-node{state}"><span class="pipeline-number">{number}</span>'
            f'<span class="pipeline-label">{html.escape(label)}</span>'
            f'<span class="pipeline-detail">{html.escape(detail)}</span></div>'
        )
    st.markdown(
        f'<div class="pipeline-scroll"><div class="pipeline">{"".join(rendered)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_question(replay: QuestionReplay) -> None:
    st.markdown(
        f'<div class="question-card"><div class="section-kicker">{html.escape(replay.category)}</div>'
        f'<div class="section-title" style="font-size:1.35rem;line-height:1.45">{html.escape(replay.question)}</div></div>',
        unsafe_allow_html=True,
    )
    for index, option in enumerate(replay.options):
        letter = chr(ord("A") + index)
        st.markdown(
            f'<div class="option-card"><span class="option-letter">{letter}</span>{html.escape(option)}</div>',
            unsafe_allow_html=True,
        )


def _render_messages(response: AgentResponseView) -> None:
    for index, message in enumerate(response.messages, start=1):
        label = f"Message {index} · {message.role.upper()}"
        with st.expander(label, expanded=index == len(response.messages)):
            st.text(message.content)
    if response.request_found_in_cache:
        st.success("Verified: the reconstructed request matches its stored cache key.")
    elif response.status == "API_ERROR":
        st.info("No cache entry exists because this API attempt returned no model reply.")
    else:
        st.warning("This reconstructed request was not found in the cache.")


def _render_response(response: AgentResponseView, *, show_change: str | None = None) -> None:
    top = st.columns([2, 1, 1, 1] if show_change is not None else [2, 1, 1])
    top[0].markdown(f"### {response.display_name}")
    top[1].metric("Answer", response.extracted_letter or "—")
    top[2].metric("Status", response.status)
    if show_change is not None:
        top[3].metric("Changed", show_change)

    st.text(response.raw_response or "No model response was returned.")
    metadata = st.columns(5)
    metadata[0].metric("Provider", response.provider)
    metadata[1].metric("Prompt tokens", f"{response.prompt_tokens:,}")
    metadata[2].metric("Completion", f"{response.completion_tokens:,}")
    metadata[3].metric("Cost", f"${response.cost_usd:.6f}")
    metadata[4].metric("Latency", f"{response.latency_seconds:.1f}s")
    st.caption(
        f"Model: {response.requested_slug} · Finish: {response.finish_reason or '—'} · Attempts: {response.attempt_count}"
    )


def _render_vote_cards(responses: tuple[AgentResponseView, ...]) -> None:
    columns = st.columns(len(responses))
    for column, response in zip(columns, responses, strict=True):
        letter = response.extracted_letter
        badge = "badge-green" if response.status == "OK" else "badge-amber"
        missing = "" if letter else " is-missing"
        with column:
            # One block, so every card shares the same baselines. Separate
            # Streamlit containers in columns do not align with each other.
            st.markdown(
                f'<div class="vote-card">'
                f'<div class="vote-agent">{html.escape(MODEL_LABELS[response.agent_id])}</div>'
                f'<div class="vote-letter{missing}">{html.escape(letter or "\u2014")}</div>'
                f'<span class="badge {badge}">{html.escape(response.status)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_vote_distribution(responses: tuple[AgentResponseView, ...]) -> None:
    counts = Counter(response.extracted_letter for response in responses if response.extracted_letter)
    rows = [{"Answer": answer, "Votes": votes} for answer, votes in sorted(counts.items())]
    if not rows:
        st.info("No valid answers were available to plot.")
        return
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=BLUE, size=38)
        .encode(
            x=alt.X("Answer:N", title="Answer option"),
            y=alt.Y("Votes:Q", title="Votes", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(tickMinStep=1)),
            tooltip=["Answer:N", "Votes:Q"],
        )
    )
    threshold = alt.Chart(alt.Data(values=[{"threshold": 3}])).mark_rule(
        color=RED, strokeDash=[6, 5], strokeWidth=2
    ).encode(y="threshold:Q")
    text = bars.mark_text(dy=-10, fontWeight=800, color="#111827").encode(text="Votes:Q")
    st.altair_chart(_base_chart((bars + threshold + text).properties(height=230)), width="stretch")
    st.caption("The dashed line is the fixed majority threshold: 3 of the 5 configured agents.")


def _render_answer_flow(replay: QuestionReplay) -> None:
    first = _response_map(replay, 1)
    second = _response_map(replay, 2)
    rows = []
    for agent_id, response in first.items():
        model = MODEL_LABELS[agent_id]
        for round_label, answer in (
            ("Round 1", response.extracted_letter),
            ("Round 2", second[agent_id].extracted_letter),
        ):
            rows.append(
                {
                    "Model": model,
                    "Round": round_label,
                    "Answer": answer or "No vote",
                }
            )
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=alt.OverlayMarkDef(size=95), strokeWidth=2.5)
        .encode(
            x=alt.X("Round:N", title=None, sort=["Round 1", "Round 2"]),
            y=alt.Y("Answer:N", title="Extracted answer", sort="descending"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(range=[BLUE, VIOLET, AMBER, GREEN, RED]),
                title=None,
            ),
            detail="Model:N",
            tooltip=["Model:N", "Round:N", "Answer:N"],
        )
        .properties(height=300)
    )
    st.altair_chart(_base_chart(chart), width="stretch")


def _render_round1_request(replay: QuestionReplay) -> None:
    st.write(
        "The same question was sent to all five agents independently. They could not see one another and were not shown a group opinion."
    )
    _render_messages(replay.round1.responses[0])
    st.markdown("#### Model settings")
    st.dataframe(
        [
            {
                "Agent": response.display_name,
                "Model": response.requested_slug,
                "Temperature": response.temperature,
                "Top-p": response.top_p,
                "Max tokens": response.max_tokens,
                "Pinned provider": response.pinned_provider or "Automatic",
                "Reasoning cap": (
                    str(response.reasoning_max_tokens)
                    if response.reasoning_max_tokens is not None
                    else "—"
                ),
            }
            for response in replay.round1.responses
        ],
        hide_index=True,
        width="stretch",
    )


def _render_round_responses(replay: QuestionReplay, round_number: int) -> None:
    responses = replay.round1.responses if round_number == 1 else replay.round2.responses
    first = _response_map(replay, 1)
    tabs = st.tabs([response.display_name for response in responses])
    for tab, response in zip(tabs, responses, strict=True):
        with tab:
            changed = None
            if round_number == 2:
                previous = first[response.agent_id].extracted_letter
                changed = "Yes" if previous != response.extracted_letter else "No"
            _render_response(response, show_change=changed)


def _render_parsing(replay: QuestionReplay) -> None:
    st.write(
        "The parser did not decide whether an answer was correct. It only checked the response format and extracted the final letter."
    )
    st.dataframe(
        [
            {
                "Agent": response.display_name,
                "Response ending": response.raw_response[-80:],
                "Parser status": response.status,
                "Extracted vote": response.extracted_letter or "No vote",
            }
            for response in replay.round1.responses
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Flow:** raw response → format check → extracted letter → vote")


def _render_round1_vote(replay: QuestionReplay) -> None:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        _render_vote_cards(replay.round1.responses)
        outcome = replay.round1.outcome
        st.markdown(
            f"### Stored vote: {outcome.consensus_state.replace('_', ' ').title()} → {outcome.consensus_answer or 'no group answer'}"
        )
        st.write(
            f"{outcome.valid_answer_count} valid answers were counted. The threshold remained three votes out of five."
        )
    with right:
        _render_vote_distribution(replay.round1.responses)

    reveal = st.toggle("Reveal benchmark answer", key="reveal_round1_answer")
    if reveal:
        if replay.round1.outcome.consensus_answer == replay.correct_answer:
            st.success(f"Correct answer: {replay.correct_answer}. The group was correct.")
        else:
            st.error(
                f"Correct answer: {replay.correct_answer}. The Round 1 group answer was not correct."
            )
    else:
        st.info("The benchmark answer is hidden until you choose to reveal it.")


def _render_round2_input(replay: QuestionReplay) -> None:
    st.write(
        "Choose an agent to inspect the exact reconstructed conversation it received. Peers are anonymous, and the Round 1 group vote is not included."
    )
    labels = {response.display_name: response for response in replay.round2.responses}
    selected_label = st.selectbox("Agent", list(labels), key="round2_agent")
    selected = labels[selected_label]

    facts = st.columns(3)
    facts[0].metric("Own valid response included", "Yes" if selected.own_round1_response_id is not None else "No")
    facts[1].metric("Anonymous peer responses", len(selected.peer_response_ids))
    facts[2].metric("Group vote included", "No")
    _render_messages(selected)


def _render_round2_responses(replay: QuestionReplay) -> None:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.dataframe(_answer_change_rows(replay), hide_index=True, width="stretch")
    with right:
        _render_answer_flow(replay)
    _render_round_responses(replay, 2)


def _render_round2_vote(replay: QuestionReplay) -> None:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        _render_vote_cards(replay.round2.responses)
        outcome = replay.round2.outcome
        st.markdown(
            f"### Stored vote: {outcome.consensus_state.replace('_', ' ').title()} → {outcome.consensus_answer or 'no group answer'}"
        )
        if outcome.consensus_answer == replay.correct_answer:
            st.success(f"Correct answer: {replay.correct_answer}. The final group answer was correct.")
        else:
            st.error(f"Correct answer: {replay.correct_answer}. The final group answer was not correct.")
    with right:
        _render_vote_distribution(replay.round2.responses)

    st.markdown("#### How the answers changed")
    st.dataframe(_answer_change_rows(replay), hide_index=True, width="stretch")


def _render_summary(replay: QuestionReplay) -> None:
    first, second = replay.round1.outcome, replay.round2.outcome
    st.markdown(
        f"## {first.consensus_state.replace('_', ' ').title()} {first.consensus_answer or '—'} → "
        f"{second.consensus_state.replace('_', ' ').title()} {second.consensus_answer or '—'}"
    )
    if first.consensus_answer != replay.correct_answer and second.consensus_answer == replay.correct_answer:
        st.success("This question changed from a wrong or missing group answer to a correct one.")
    elif first.consensus_answer == replay.correct_answer and second.consensus_answer != replay.correct_answer:
        st.error("This question regressed: the Round 1 group answer was correct, but the final answer was not.")

    totals = _total_usage(replay)
    columns = st.columns(4)
    columns[0].metric("Total tokens", f"{totals['prompt_tokens'] + totals['completion_tokens']:,}")
    columns[1].metric("Prompt tokens", f"{totals['prompt_tokens']:,}")
    columns[2].metric("Total cost", f"${totals['cost_usd']:.6f}")
    columns[3].metric("Stored latency", f"{totals['latency_seconds']:.1f}s")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("#### Round comparison")
        st.dataframe(
            [
                {
                    "Round": 1,
                    "State": first.consensus_state,
                    "Group answer": first.consensus_answer or "No answer",
                    "Correct": first.consensus_answer == replay.correct_answer,
                    "Valid votes": first.valid_answer_count,
                    "Cost": f"${first.total_cost_usd:.6f}",
                },
                {
                    "Round": 2,
                    "State": second.consensus_state,
                    "Group answer": second.consensus_answer or "No answer",
                    "Correct": second.consensus_answer == replay.correct_answer,
                    "Valid votes": second.valid_answer_count,
                    "Cost": f"${second.total_cost_usd:.6f}",
                },
            ],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("#### Answer movement")
        _render_answer_flow(replay)


def _render_step(step_name: str, replay: QuestionReplay) -> None:
    renderers = {
        "question": _render_question,
        "round1_request": _render_round1_request,
        "round1_responses": lambda item: _render_round_responses(item, 1),
        "parsing": _render_parsing,
        "round1_vote": _render_round1_vote,
        "round2_input": _render_round2_input,
        "round2_responses": _render_round2_responses,
        "round2_vote": _render_round2_vote,
        "summary": _render_summary,
    }
    try:
        renderers[step_name](replay)
    except KeyError as error:
        raise ValueError(f"unknown replay step {step_name!r}") from error


def _render_replay(replay: QuestionReplay, example_label: str) -> None:
    _page_header(
        "Interactive evidence replay",
        "Follow one real debate from prompt to final vote.",
        "Inspect the exact stored responses, reconstructed requests, parsing decisions, anonymous peer exchange and answer movement.",
    )

    first, second = replay.round1.outcome, replay.round2.outcome
    final_correct = second.consensus_answer == replay.correct_answer
    _metric_row((
        ("Question", replay.question_id.split(":")[-1], example_label, "blue"),
        ("Round 1", first.consensus_answer or "No answer", first.consensus_state.replace("_", " ").title(), "amber"),
        ("Round 2", second.consensus_answer or "No answer", second.consensus_state.replace("_", " ").title(), "violet"),
        ("Final outcome", "Correct" if final_correct else "Not correct",
         f"Benchmark answer {replay.correct_answer}", "green" if final_correct else "amber"),
    ))

    if "replay_step" not in st.session_state:
        st.session_state.replay_step = 0
    step_index = min(max(int(st.session_state.replay_step), 0), len(STEPS) - 1)
    step_name, step_title = STEPS[step_index]

    st.write("")
    _render_pipeline(step_index)
    st.progress((step_index + 1) / len(STEPS))
    st.markdown(f"## Step {step_index + 1}: {step_title}")
    st.caption(f"Subject: {replay.category.title()} · Stored run: {replay.run_id}")
    _render_step(step_name, replay)
    st.divider()
    _render_navigation(step_index)


def main() -> None:
    st.set_page_config(page_title="Multi-agent debate replay", page_icon="▤", layout="wide")
    _inject_styles()

    with st.sidebar:
        _brand()
        page = st.radio("Workspace", ("Overview", "Debate replay"), label_visibility="collapsed")
        selected_label = next(iter(CURATED_QUESTIONS))
        if page == "Debate replay":
            st.divider()
            st.markdown("#### Curated evidence")
            selected_label = st.selectbox("Choose a case", tuple(CURATED_QUESTIONS))
            st.caption(CURATED_NOTES[CURATED_QUESTIONS[selected_label]])
        st.divider()
        st.markdown('<span class="badge badge-green">● Read-only mode</span>', unsafe_allow_html=True)
        st.caption("No model calls · no database writes")
        st.caption("Accepted run · agents_v7")

    try:
        overview = _load_overview()
        if page == "Overview":
            _render_overview(overview)
        else:
            question_id = CURATED_QUESTIONS[selected_label]
            replay = _load_replay(question_id)
            _render_replay(replay, selected_label)
    except (ViewerDataError, OSError, sqlite3.Error, json.JSONDecodeError) as error:
        st.error(f"The stored experiment could not be loaded: {error}")
        st.stop()


if __name__ == "__main__":
    main()
