import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_groq_client():
    """Initializes and returns the Groq client using Streamlit secrets or environment variables."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        return None
    
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


class LLMConnector:
    """Manages Groq LLM API interaction, memory context injection, and intelligent fact processing."""

    def __init__(self):
        self.client = get_groq_client()

    def is_configured(self) -> bool:
        """Returns True if the Groq client is initialized."""
        return self.client is not None

    def fetch_models(self) -> list[str]:
        """Fetches available Groq models or returns recommended defaults."""
        if not self.is_configured():
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

        try:
            resp = self.client.models.list()
            all_ids = sorted([m.id for m in resp.data])
            priority = [m for m in all_ids if any(x in m for x in ["70b", "versatile", "32b"])]
            rest = [m for m in all_ids if m not in priority]
            return priority + rest if (priority or rest) else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        except Exception:
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

    def build_system_prompt(self, memory_facts: list[str]) -> str:
        """Constructs system prompt with injected persistent memory facts."""
        base_prompt = (
            "You are a helpful AI assistant equipped with long-term persistent memory. "
            "You recall details about the user across past conversations."
        )

        if not memory_facts:
            return base_prompt

        facts_block = "\n".join(f"- {fact}" for fact in memory_facts)
        return (
            f"{base_prompt}\n\n"
            "Below are known facts and preferences about the user retrieved from persistent memory:\n"
            f"{facts_block}\n\n"
            "Guidelines:\n"
            "1. Seamlessly use these memory facts to provide personalized, accurate responses.\n"
            "2. Do not explicitly state 'According to my database' or 'My memory says' unless asked.\n"
            "3. Speak naturally as a human friend or assistant who remembers previous conversations."
        )

    def extract_user_facts(self, user_message: str) -> list[str]:
        """Intelligently extracts durable personal facts/preferences/status about the user from a prompt."""
        if not self.is_configured() or not user_message:
            return []

        system_prompt = (
            "You are an expert fact extractor. Analyze the user's message and extract durable, long-term personal facts, "
            "preferences, background details, or status updates about the user.\n"
            "Return ONLY a valid JSON object with the schema:\n"
            '{\n  "facts": ["User preference or status fact 1", "User fact 2"]\n}\n'
            "Rules:\n"
            "1. Ignore greetings, general queries, general knowledge questions, and conversational chatter.\n"
            "2. Only extract durable facts explicitly stated by the user about themselves.\n"
            "3. If no durable user facts exist, return {\"facts\": []}."
        )

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            return data.get("facts", [])
        except Exception:
            return []

    def find_obsolete_memory_ids(self, new_fact: str, existing_memories: list[dict]) -> list[str]:
        """Identifies existing memory IDs that are contradicted or rendered obsolete by a new fact."""
        if not self.is_configured() or not new_fact or not existing_memories:
            return []

        system_prompt = (
            "You are a memory manager. Given a new user fact and a list of existing stored memory items (each with 'id' and 'memory'), "
            "identify existing memory items that are contradicted, outdated, or superseded by the new fact.\n"
            "Return ONLY a valid JSON object with schema:\n"
            '{\n  "obsolete_ids": ["mem_id_1"]\n}\n'
            "If no existing memory is contradicted or outdated, return {\"obsolete_ids\": []}."
        )

        payload = {
            "new_fact": new_fact,
            "existing_memories": [
                {"id": item.get("id"), "memory": item.get("memory")}
                for item in existing_memories if isinstance(item, dict) and item.get("id") and item.get("memory")
            ]
        }

        if not payload["existing_memories"]:
            return []

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            return data.get("obsolete_ids", [])
        except Exception:
            return []

    def stream_chat_completion(self, model: str, chat_history: list[dict], memory_facts: list[str]):
        """Creates a streaming response generator from Groq API."""
        if not self.is_configured():
            yield "GROQ_API_KEY is not configured. Please check your settings."
            return

        system_content = self.build_system_prompt(memory_facts)
        api_messages = [{"role": "system", "content": system_content}] + chat_history

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit" in err_msg:
                yield "Rate limit exceeded. Please wait a moment before trying again."
            elif "api key" in err_msg or "authentication" in err_msg:
                yield "Invalid API Key. Please verify your Groq API credentials."
            else:
                yield f"API Error: {str(e)}"
