from uuid import uuid4

import pandas as pd
import streamlit as st

from arxiv_researcher.agent.main import ArxivResearcher
from arxiv_researcher.settings import settings
from arxiv_researcher.ui.types import (
    AlertMessage,
    ChatMessage,
    DataframeMessage,
    Message,
    SearchProgress,
)


def show_chat_message(message: Message | None):
    if not message:
        return
    chat_message: ChatMessage = message.content
    with st.chat_message(chat_message.role):
        st.markdown(chat_message.content)


def show_dataframe_message(message: Message | None):
    if not message:
        return
    dataframe_message: DataframeMessage = message.content
    df = pd.DataFrame(dataframe_message.data)
    with st.chat_message(dataframe_message.role):
        st.markdown(dataframe_message.content)
        st.dataframe(df)


def show_alert_message(message: Message | None):
    if not message:
        return
    alert_message: AlertMessage = message.content
    with st.chat_message(alert_message.role):
        st.success(alert_message.content)


def show_search_progress(message: Message | None):
    if not message:
        return
    search_progress: SearchProgress = message.content
    with st.chat_message(search_progress.role):
        with st.expander("Search Progress", expanded=False):
            st.markdown(search_progress.content)


def init_session_state():
    if "researcher" not in st.session_state:
        researcher = ArxivResearcher(llm=settings.llm)
        researcher.subscribe("chat_message", show_chat_message)
        researcher.subscribe("dataframe_message", show_dataframe_message)
        researcher.subscribe("alert_message", show_alert_message)
        researcher.subscribe("search_progress", show_search_progress)
        st.session_state.researcher = researcher
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex


def set_page_config():
    st.set_page_config(
        page_title="arXiv Researcher",
        page_icon="🧊",
        layout="wide",
    )


def display_sidebar():
    with st.sidebar:
        st.image(st.session_state.researcher.mermaid_png())
        if st.sidebar.button(
            "Clear Chat History", key="clear", type="primary", use_container_width=True
        ):
            st.session_state.researcher.reset()
            st.session_state.thread_id = uuid4().hex
            st.rerun()


def display_header():
    st.title("arXiv Researcher")


def process_query(query: str):
    st.session_state.researcher.handle_human_message(query, st.session_state.thread_id)


def main():
    init_session_state()
    set_page_config()
    display_sidebar()
    display_header()

    query = st.chat_input("What do you want to know? I will give you an answer.")
    if query:
        with st.spinner("Processing..."):
            process_query(query)
