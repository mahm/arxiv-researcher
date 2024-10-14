from uuid import uuid4

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

from arxiv_researcher.agent.arxiv_researcher import ArxivResearcher
from arxiv_researcher.settings import settings
from arxiv_researcher.ui.types import (
    AlertMessage,
    ChatMessage,
    DataframeMessage,
    Message,
)


def show_message(message: Message | None):
    if not message:
        return
    if isinstance(message.content, ChatMessage):
        chat_message: ChatMessage = message.content
        with st.chat_message(chat_message.role):
            st.markdown(chat_message.content)
    elif isinstance(message.content, DataframeMessage):
        dataframe_message: DataframeMessage = message.content
        df = pd.DataFrame(dataframe_message.data)
        with st.chat_message(dataframe_message.role):
            st.markdown(dataframe_message.content)
            st.dataframe(df)
    elif isinstance(message.content, AlertMessage):
        alert_message: AlertMessage = message.content
        with st.chat_message(alert_message.role):
            st.success(alert_message.content)
    else:
        raise ValueError(f"Invalid message type: {type(message.content)}")


def init_session_state():
    if "researcher" not in st.session_state:
        researcher = ArxivResearcher(llm=settings.llm)
        researcher.subscribe(show_message)
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
        with st.spinner("Searching..."):
            process_query(query)
