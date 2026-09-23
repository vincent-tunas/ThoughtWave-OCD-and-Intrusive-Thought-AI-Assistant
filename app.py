from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from thoughtwave.config import Settings, load_settings
from thoughtwave.database import Database
from thoughtwave.detector import BehavioralStateDetector, DetectorConfig
from thoughtwave.embeddings import SemanticMemory, SentenceTransformerEncoder
from thoughtwave.models import BehavioralState
from thoughtwave.openrouter import OpenRouterClient, OpenRouterError
from thoughtwave.service import ThoughtWaveService


st.set_page_config(
    page_title="ThoughtWave",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0e1117;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #161b22;
    }

    .tw-hero {
        padding: 1rem 0 0.5rem;
    }

    .tw-hero h1 {
        color: #f8fafc;
        font-size: 2.25rem;
        margin: 0;
    }

    .tw-hero p {
        color: #cbd5e1;
        max-width: 46rem;
        margin-top: 0.35rem;
    }

    .tw-note {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.4;
    }

    .tw-state {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .tw-normal {
        background: #dcfce7;
        color: #166534;
    }

    .tw-monitor {
        background: #dbeafe;
        color: #1e40af;
    }

    .tw-possible_loop {
        background: #fef3c7;
        color: #92400e;
    }

    .tw-strong_loop {
        background: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class Runtime:
    settings: Settings
    database: Database
    memory: SemanticMemory
    service: ThoughtWaveService
    indexed_messages: int


@st.cache_resource(show_spinner="Loading the local semantic model…")
def initialize_runtime() -> Runtime:
    settings = load_settings()
    if settings.missing_required_values:
        joined = ", ".join(settings.missing_required_values)
        raise RuntimeError(f"Missing required environment values: {joined}")

    database = Database(settings.database_path)
    database.initialize()
    encoder = SentenceTransformerEncoder(
        settings.hf_embedding_model, settings.hf_token
    )
    memory = SemanticMemory(encoder)
    indexed_messages = memory.rebuild(database)
    detector = BehavioralStateDetector(
        DetectorConfig(similarity_threshold=settings.similarity_threshold)
    )
    llm = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout_seconds=settings.openrouter_timeout_seconds,
        max_retries=settings.openrouter_max_retries,
        app_url=settings.app_url,
    )
    service = ThoughtWaveService(
        database=database,
        semantic_memory=memory,
        detector=detector,
        llm=llm,
        top_k=settings.top_k_related_messages,
        recent_history_limit=settings.recent_history_limit,
    )
    return Runtime(settings, database, memory, service, indexed_messages)


def state_badge(state: str) -> str:
    css_name = state.lower()
    label = state.replace("_", " ").title()
    return f'<span class="tw-state tw-{css_name}">{html.escape(label)}</span>'


def select_or_create_session(database: Database) -> str:
    sessions = database.list_sessions()
    if "session_id" not in st.session_state:
        st.session_state.session_id = sessions[0]["id"] if sessions else database.create_session()

    if st.sidebar.button("＋ New conversation", use_container_width=True):
        st.session_state.session_id = database.create_session()
        st.rerun()

    sessions = database.list_sessions()
    session_ids = [item["id"] for item in sessions]
    if st.session_state.session_id not in session_ids:
        st.session_state.session_id = database.create_session()
        sessions = database.list_sessions()
        session_ids = [item["id"] for item in sessions]

    title_by_id = {item["id"]: item["title"] for item in sessions}
    selected = st.sidebar.selectbox(
        "Conversation",
        session_ids,
        index=session_ids.index(st.session_state.session_id),
        format_func=lambda item: title_by_id[item],
    )
    st.session_state.session_id = selected
    return selected


def render_sidebar(runtime: Runtime, session_id: str) -> bool:
    st.sidebar.markdown("### System status")
    st.sidebar.success("Embedding model ready")
    st.sidebar.success("OpenRouter configured")
    st.sidebar.success("SQLite ready")
    st.sidebar.caption(
        f"Model: {runtime.settings.openrouter_model}\n\n"
        f"Semantic messages indexed: {runtime.indexed_messages}"
    )
    show_debug = st.sidebar.toggle("Show detector details", value=False)
    st.sidebar.divider()
    if st.sidebar.button("Delete this conversation", use_container_width=True):
        runtime.database.delete_session(session_id)
        runtime.memory.remove_session(session_id)
        st.session_state.pop("session_id", None)
        st.rerun()
    st.sidebar.markdown(
        '<p class="tw-note">ThoughtWave is a research and portfolio prototype. '
        "It does not diagnose, provide therapy, or replace emergency support.</p>",
        unsafe_allow_html=True,
    )
    return show_debug


st.markdown(
    """
    <div class="tw-hero">
      <h1>ThoughtWave</h1>
      <p>A conversation assistant that answers normally and adapts when repeated
      certainty seeking becomes more likely.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    runtime = initialize_runtime()
except Exception as exc:
    st.error("ThoughtWave could not finish initialization.")
    st.code(str(exc))
    st.info(
        "Copy .env.example to .env, add HF_TOKEN and OPENROUTER_API_KEY, "
        "choose an OPENROUTER_MODEL, then restart Streamlit."
    )
    st.stop()

session_id = select_or_create_session(runtime.database)
show_debug = render_sidebar(runtime, session_id)
messages = runtime.database.get_messages(session_id)

if not messages:
    st.info("Start with any question. ThoughtWave begins in normal-answer mode.")

for message in messages:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if show_debug and message.role == "assistant" and message.behavioral_state:
            st.markdown(state_badge(message.behavioral_state), unsafe_allow_html=True)
            if message.detector_metadata:
                with st.expander("Detector evidence"):
                    st.json(message.detector_metadata)

user_text = st.chat_input("What is on your mind?")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = runtime.service.respond(session_id, user_text)
            st.markdown(result.answer)
            if show_debug:
                st.markdown(
                    state_badge(result.detection.state.value),
                    unsafe_allow_html=True,
                )
                with st.expander("Detector evidence"):
                    st.json(result.detection.to_metadata())
                    st.caption(
                        f"OpenRouter model: {result.generation.model} | "
                        f"Request: {result.generation.request_id or 'not supplied'}"
                    )
    except OpenRouterError as exc:
        st.error(str(exc))
        st.caption(
            "No fallback answer was displayed because a valid OpenRouter completion "
            "was not received."
        )
    except Exception as exc:
        st.error(f"ThoughtWave could not process this message: {exc}")

