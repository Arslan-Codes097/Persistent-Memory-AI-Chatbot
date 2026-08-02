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
    """Manages fact storage, retrieval, intelligent deduplication, and deletion using standard Mem0 v2 API specifications."""

    def __init__(self):
        self.client = get_mem0_client()

    def is_configured(self) -> bool:
        """Returns True if the Mem0 client is successfully initialized."""
        return self.client is not None

    def add_interaction(self, user_id: str, user_message: str, assistant_message: str, llm_connector=None) -> bool:
        """Intelligently extracts core user facts, overwrites outdated memories, and stores clean statements in Mem0."""
        if not self.is_configured() or not user_id:
            return False
        
        # Step 1: Use LLM connector to extract durable user facts (ignoring chatter & greetings)
        if llm_connector and hasattr(llm_connector, "extract_user_facts"):
            extracted_facts = llm_connector.extract_user_facts(user_message)
        else:
            extracted_facts = [user_message]

        if not extracted_facts:
            # Skip storing if prompt contains no durable personal facts
            return True

        existing_memories = self.get_all_memories(user_id)

        # Step 2: Process each extracted fact
        for fact in extracted_facts:
            # Find and delete any existing memories contradicted or rendered obsolete by this new fact
            if llm_connector and hasattr(llm_connector, "find_obsolete_memory_ids") and existing_memories:
                obsolete_ids = llm_connector.find_obsolete_memory_ids(fact, existing_memories)
                for mem_id in obsolete_ids:
                    self.delete_single_memory(mem_id)

            # Add the clean, single-fact string to Mem0
            try:
                self.client.add(fact, user_id=user_id)
            except Exception as e:
                st.error(f"Error adding memory fact: {e}")

        return True

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

    def delete_single_memory(self, memory_id: str) -> bool:
        """Deletes a specific memory item by memory_id."""
        if not self.is_configured() or not memory_id:
            return False

        try:
            self.client.delete(memory_id=memory_id)
            return True
        except Exception as e:
            st.error(f"Failed to delete memory item: {e}")
            return False

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
