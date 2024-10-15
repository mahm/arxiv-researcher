from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ExpandMessage(BaseModel):
    role: Literal["user", "assistant"]
    title: str
    content: str | list[dict]


class DataframeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    data: list[Any]


class AlertMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SearchProgress(BaseModel):
    role: Literal["user", "assistant"]
    task: str
    content: str


class Message(BaseModel):
    is_need_human_feedback: bool = False
    is_done: bool = False
    content: (
        ChatMessage | ExpandMessage | DataframeMessage | AlertMessage | SearchProgress
    )
