from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DataframeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    data: list[Any]


class AlertMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Message(BaseModel):
    content: ChatMessage | DataframeMessage | AlertMessage
