import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_groq_client():
    """Initializes and caches the Groq API client using Streamlit secrets or environment variables."""
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
    """Manages Groq LLM API interaction, memory context injection, intelligent fact extraction, and topic-isolated memory target resolution."""

    def __init__(self):
        self.client = get_groq_client()

    def is_configured(self) -> bool:
        """Returns True if the Groq client is initialized."""
        return self.client is not None

    @st.cache_data(ttl=600)
    def fetch_models(_self) -> list[str]:
        """Fetches available Groq models with 10-minute caching to optimize UI performance."""
        if not _self.is_configured():
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

        try:
            resp = _self.client.models.list()
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
        """Intelligently extracts durable personal facts, preferences, food likes, education, and status updates."""
        if not self.is_configured() or not user_message:
            return []

        system_prompt = (
            "You are an intelligent memory fact extractor. Extract all durable personal facts, user preferences, "
            "food/drink likes/dislikes, hobbies, background, education, job status, or life updates stated by the user.\n"
            "Return ONLY a valid JSON object with schema:\n"
            '{\n  "facts": ["User fact 1", "User fact 2"]\n}\n'
            "Rules:\n"
            "1. DO include personal preferences (e.g. 'I love mangoes' -> 'User loves mangoes').\n"
            "2. DO include status updates (e.g. 'I completed my CS degree' -> 'User completed CS degree').\n"
            "3. DO NOT extrapolate or alter status beyond what was stated.\n"
            "4. Ignore generic greetings ('hello'), bot commands, and general knowledge questions.\n"
            "5. Format each fact as a clear third-person statement starting with 'User...' or 'Name: ...'.\n"
            "6. If no durable personal facts exist, return {\"facts\": []}."
        )

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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

    def find_target_memory_to_update(self, new_fact: str, existing_memories: list[dict]) -> str | None:
        """Determines if a new fact is a direct status update or replacement for an existing memory on the exact same topic."""
        if not self.is_configured() or not new_fact or not existing_memories:
            return None

        system_prompt = (
            "You are an expert memory topic classifier and update resolver.\n"
            "Your job is to determine if a new user fact is a direct STATUS UPDATE or REPLACEMENT for an existing memory.\n"
            "STRICT RULE: An update_id MUST ONLY be returned if the new fact and existing memory belong to the EXACT SAME TOPIC CATEGORY.\n\n"
            "Topic Categories:\n"
            "- Name: (e.g. 'Name: Bilal Babar' replaces 'Name: Arslan Babar') -> MATCH\n"
            "- Education: (e.g. 'User completed CS degree' replaces 'User pursuing CS degree') -> MATCH\n"
            "- Work/Internship: (e.g. 'User completed ML internship' replaces 'User interning in ML') -> MATCH\n"
            "- Preferences/Food: (e.g. 'User loves pizza' vs 'User loves mangoes') -> DO NOT MATCH (return null, keep both!)\n\n"
            "CROSS-TOPIC RULE (CRITICAL):\n"
            "- Work/Internship does NOT replace Education (e.g. 'Doing ML internship' does NOT replace 'Pursuing CS degree at UET'). Return null!\n"
            "- Food/Hobbies does NOT replace Education or Work. Return null!\n\n"
            "Return ONLY JSON: {\"update_id\": \"mem_id_here_or_null\"}"
        )

        payload = {
            "new_fact": new_fact,
            "existing_memories": [
                {"id": item.get("id"), "memory": item.get("memory")}
                for item in existing_memories if isinstance(item, dict) and item.get("id") and item.get("memory")
            ]
        }

        if not payload["existing_memories"]:
            return None

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            res_id = data.get("update_id")
            return res_id if res_id and res_id != "null" and res_id != "None" else None
        except Exception:
            return None

    def stream_chat_completion(self, model: str, chat_history: list[dict], memory_facts: list[str]):
        """Creates a streaming response generator from Groq API."""
        if not self.is_configured():
            raise RuntimeError("GROQ_API_KEY is not configured. Please check your settings.")

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
                raise RuntimeError("Rate limit exceeded. Please wait a moment before trying again.")
            elif "api key" in err_msg or "authentication" in err_msg:
                raise RuntimeError("Invalid API Key. Please verify your Groq API credentials.")
            else:
                raise RuntimeError(f"API Error: {str(e)}")
