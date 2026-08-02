import streamlit as st
from memory_handler import MemoryHandler
from llm_connector import LLMConnector

st.set_page_config(
    page_title="Persistent Memory AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize cached singletons
memory_handler = MemoryHandler()
llm_connector = LLMConnector()


# ── Sidebar Setup ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Persistent AI Assistant")
    st.caption("Powered by Groq & Mem0 Cloud Platform")
    st.divider()

    st.subheader("User Profile")
    user_name_input = st.text_input(
        "Your Name",
        value=st.session_state.get("user_name_display", "Arslan"),
        help="Memories are isolated per user. Entering your name creates or loads your personalized memory store.",
    ).strip()

    # Data validation & slug normalization for Mem0 API user_id
    if not user_name_input:
        user_name_display = "Guest"
    else:
        user_name_display = user_name_input

    # Normalize name to clean slug for Mem0 backend storage (e.g. "Arslan Ali" -> "arslan_ali")
    user_id = "".join(c if c.isalnum() else "_" for c in user_name_display.lower()).strip("_")
    if not user_id:
        user_id = "guest_user"

    st.session_state.user_name_display = user_name_display

    # Reset chat history if user switches identity
    if "current_user_id" not in st.session_state or st.session_state.current_user_id != user_id:
        st.session_state.current_user_id = user_id
        st.session_state.messages = []
        st.session_state.last_injected_memories = []

    st.divider()

    st.subheader("AI Model Configuration")
    available_models = llm_connector.fetch_models()
    selected_model = st.selectbox(
        "Select Groq Model",
        available_models,
        index=0,
    )

    st.divider()

    # Clear session conversation history (local UI reset)
    if st.button("New Session", use_container_width=True, help="Clear active chat window while preserving saved memories in Mem0."):
        st.session_state.messages = []
        st.session_state.last_injected_memories = []
        st.rerun()

    st.divider()

    # Live Memory Inspector
    st.subheader("Memory Store Inspector")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True, help="Re-fetch stored facts from Mem0 Cloud to see newly extracted memories."):
            st.rerun()
    with col2:
        if st.button("Clear All", use_container_width=True, help="Permanently delete all saved memory facts for this user from Mem0."):
            if memory_handler.delete_all_memories(user_id):
                st.success(f"Cleared all stored memories for '{user_name_display}'.")
                st.session_state.last_injected_memories = []
                st.rerun()

    stored_memories = memory_handler.get_all_memories(user_id)
    
    if stored_memories:
        st.caption(f"Facts remembered for '{user_name_display}': {len(stored_memories)}")
        with st.expander("View & Manage Stored Facts", expanded=True):
            for idx, item in enumerate(stored_memories):
                mem_id = item.get("id") if isinstance(item, dict) else str(idx)
                mem_text = item.get("memory") if isinstance(item, dict) else str(item)

                with st.container(border=True):
                    st.markdown(f"**Fact #{idx + 1}**")
                    st.write(mem_text)
                    if st.button("🗑️ Delete", key=f"del_fact_{mem_id}_{idx}", use_container_width=True):
                        if memory_handler.delete_single_memory(mem_id):
                            st.toast("Memory deleted successfully!", icon="🗑️")
                            st.session_state.last_injected_memories = []
                            st.rerun()
    else:
        st.info(f"No persistent facts recorded yet for '{user_name_display}'.")


# ── Session State Initialization ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_injected_memories" not in st.session_state:
    st.session_state.last_injected_memories = []


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("Persistent Memory AI Chatbot")
st.caption(f"An intelligent companion that continuously learns, remembers **{user_name_display}**, and adapts to your preferences across every session.")
st.divider()

# Check configuration warnings
if not memory_handler.is_configured():
    st.warning("MEM0_API_KEY is not set. Memory persistence will be inactive. Add MEM0_API_KEY to your .env or secrets.")

if not llm_connector.is_configured():
    st.error("GROQ_API_KEY is missing. Please add GROQ_API_KEY to your .env or Streamlit secrets.")
    st.stop()


# Render previous conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat Input Processing ─────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message here..."):

    # Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Retrieve relevant persistent memories for this prompt
    relevant_memories = memory_handler.search_memories(user_id, prompt)
    st.session_state.last_injected_memories = relevant_memories

    # Step 2: Render Assistant Message with streamed response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        # Prepare messages array for API
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        try:
            # Stream response from LLM with memory context
            stream_gen = llm_connector.stream_chat_completion(
                model=selected_model,
                chat_history=chat_history,
                memory_facts=relevant_memories,
            )

            for chunk in stream_gen:
                full_reply += chunk
                placeholder.markdown(full_reply + "▌")
            
            placeholder.markdown(full_reply)

            # Step 3: Append response to session chat history
            st.session_state.messages.append({"role": "assistant", "content": full_reply})

            # Step 4: Send conversation turn to Mem0 for intelligent extraction & deduplication
            if full_reply:
                with st.spinner("Processing persistent memory..."):
                    memory_handler.add_interaction(
                        user_id=user_id,
                        user_message=prompt,
                        assistant_message=full_reply,
                        llm_connector=llm_connector,
                        existing_memories=stored_memories
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")
