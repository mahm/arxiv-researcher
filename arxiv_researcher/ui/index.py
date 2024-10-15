from enum import Enum, auto
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


class AppState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    WAITING_FOR_USER = auto()


def show_message(message: Message | None):
    if not message:
        return
    content = message.content
    with st.chat_message(content.role):
        if isinstance(content, ChatMessage):
            st.markdown(content.content)
        elif isinstance(content, DataframeMessage):
            st.markdown(content.content)
            st.dataframe(pd.DataFrame(content.data))
        elif isinstance(content, AlertMessage):
            st.success(content.content)
        elif isinstance(content, SearchProgress):
            with st.expander("Search Progress", expanded=False):
                st.markdown(content.content)


def init_session_state():
    if "researcher" not in st.session_state:
        researcher = ArxivResearcher(llm=settings.llm)
        st.session_state.researcher = researcher
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stream_generator" not in st.session_state:
        st.session_state.stream_generator = None
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState.IDLE


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
            st.session_state.messages = []
            st.session_state.stream_generator = None
            st.session_state.app_state = AppState.IDLE
            st.rerun()


def display_header():
    st.title("arXiv Researcher")


def process_stream():
    if st.session_state.stream_generator is None:
        return

    try:
        with st.spinner("Processing..."):
            message = next(st.session_state.stream_generator)
            st.session_state.messages.append(message)
            show_message(message)

            if message.is_done:
                st.session_state.stream_generator = None
                st.session_state.app_state = AppState.IDLE
            elif message.is_need_human_feedback:
                st.session_state.app_state = AppState.WAITING_FOR_USER
            st.rerun()
    except StopIteration:
        st.session_state.stream_generator = None
        st.session_state.app_state = AppState.IDLE


def main():
    init_session_state()
    set_page_config()
    display_sidebar()
    display_header()

    for message in st.session_state.messages:
        show_message(message)

    query = st.chat_input("What do you want to know? I will give you an answer.")

    if query and st.session_state.app_state != AppState.PROCESSING:
        st.session_state.messages.append(
            Message(content=ChatMessage(role="user", content=query))
        )
        show_message(st.session_state.messages[-1])
        st.session_state.stream_generator = (
            st.session_state.researcher.handle_human_message(
                query=query,
                thread_id=st.session_state.thread_id,
            )
        )
        st.session_state.app_state = AppState.PROCESSING
        st.rerun()

    if st.session_state.app_state == AppState.PROCESSING:
        process_stream()


if __name__ == "__main__":
    main()
