import os
import streamlit as st
from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()


def get_mem0_client():
    """Initializes and returns the Mem0 MemoryClient using Streamlit secrets or environment variables."""
    try:
        api_key = st.secrets["MEM0_API_KEY"]
    except Exception:
        api_key = os.getenv("MEM0_API_KEY", "")

    if not api_key:
        return None
    
    try:
        return MemoryClient(api_key=api_key)
    except Exception:
        return None


class MemoryHandler:
    """Manages fact storage, retrieval, and deletion using standard Mem0 v2 API specifications."""

    def __init__(self):
        self.client = get_mem0_client()

    def is_configured(self) -> bool:
        """Returns True if the Mem0 client is successfully initialized."""
        return self.client is not None

    def add_interaction(self, user_id: str, user_message: str, assistant_message: str) -> bool:
        """Sends a conversation turn to Mem0 for automatic fact extraction and semantic update."""
        if not self.is_configured() or not user_id:
            return False
        
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        try:
            self.client.add(messages, user_id=user_id)
            return True
        except Exception as e:
            st.error(f"Error updating memory: {e}")
            return False

    def search_memories(self, user_id: str, query: str) -> list[str]:
        """Retrieves relevant facts from Mem0 for context injection using standard Mem0 v2 filters."""
        if not self.is_configured() or not user_id or not query:
            return []

        try:
            response = self.client.search(query, filters={"user_id": user_id})
            return self._parse_memory_texts(response)
        except Exception as e:
            st.warning(f"Unable to search memories: {e}")
            return []

    def get_all_memories(self, user_id: str) -> list[dict]:
        """Fetches all stored memories for a specific user using standard Mem0 v2 filters."""
        if not self.is_configured() or not user_id:
            return []

        try:
            response = self.client.get_all(filters={"user_id": user_id})
            if isinstance(response, dict) and "results" in response:
                return response["results"]
            elif isinstance(response, list):
                return response
            return []
        except Exception as e:
            st.warning(f"Unable to retrieve memories: {e}")
            return []

    def delete_all_memories(self, user_id: str) -> bool:
        """Deletes all stored memories for a specific user ID."""
        if not self.is_configured() or not user_id:
            return False

        try:
            self.client.delete_all(user_id=user_id)
            return True
        except Exception as e:
            st.error(f"Failed to clear memories: {e}")
            return False

    def _parse_memory_texts(self, response) -> list[str]:
        """Helper to extract clean memory text strings from Mem0 response objects."""
        memories = []
        items = response.get("results", []) if isinstance(response, dict) else response
        
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    memory_str = item.get("memory") or item.get("text") or item.get("content")
                    if memory_str:
                        memories.append(str(memory_str))
                elif isinstance(item, str):
                    memories.append(item)
        return memories
