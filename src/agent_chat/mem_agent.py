from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import Message 


class ChatState(BaseModel):
    """State manager for chat history."""
    messages: List[Message] = Field(
        default_factory=list,
        description="Chat conversation history",
        max_length=1000  # Prevent memory issues
    )
    
    def append_messages(self, new_messages: List[Message]) -> None:
        """Add new messages to history."""
        self.messages.extend(new_messages)
        if len(self.messages) > 1000:
            # Keep most recent messages if we exceed the limit
            self.messages = self.messages[-1000:]
    
    def clear_messages(self) -> None:
        """Reset chat history."""
        self.messages.clear()
