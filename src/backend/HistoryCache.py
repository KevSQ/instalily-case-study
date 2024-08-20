from typing import List, Final
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain.pydantic_v1 import BaseModel, Field


class HistoryCache(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history, ideal for demoing and fast iteration. """

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


STORE = {}


def get_session_history( session_id: str) -> BaseChatMessageHistory:
    if (session_id) not in STORE:
        STORE[(session_id)] = HistoryCache()
    return STORE[(session_id)]


def get_user_id():
    return '1'


def get_session_id():
    return '1'

def get_store():
    return STORE