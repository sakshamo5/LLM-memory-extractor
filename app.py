import streamlit as st
from memory_extractor import MemoryExtractor
from personality_engine import PersonalityEngine
from config import GROQ_API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS
import json
from datetime import datetime

st.set_page_config(
    page_title="Companion AI - Memory & Personality",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .json-display {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .personality-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.memory_extractor = MemoryExtractor(GROQ_API_KEY)
        st.session_state.personality_engine = PersonalityEngine(GROQ_API_KEY)
        st.session_state.chat_history = []
        st.session_state.extracted_memory = None
        st.session_state.current_model = DEFAULT_MODEL
        st.session_state.current_personality = "Neutral Assistant"
        st.session_state.message_count = 0
        st.session_state.memory_extracted = False
        st.session_state.initialized = True

initialize_session_state()

with st.sidebar:
    st.title("🧠 Companion AI Settings")
    
    st.subheader("Model Configuration")
    selected_model_name = st.selectbox(
        "Select LLM Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(st.session_state.current_model)
    )
    
    selected_model = AVAILABLE_MODELS[selected_model_name]
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.memory_extractor.update_model(selected_model)
        st.session_state.personality_engine.update_model(selected_model)
        st.success(f"Switched to {selected_model_name}")
    
    st.divider()
    
    st.subheader("Personality Selection")
    personalities = [
        "Neutral Assistant",
        "Calm Mentor",
        "Witty Friend",
        "Therapist",
        "Enthusiastic Cheerleader",
        "Philosophical Sage",
        "Technical Expert"
    ]
    
    selected_personality = st.selectbox(
        "Choose Agent Personality",
        options=personalities,
        index=personalities.index(st.session_state.current_personality)
    )
    
    if selected_personality != st.session_state.current_personality:
        st.session_state.current_personality = selected_personality
        st.success(f"Personality: {selected_personality}")
    
    st.divider()
    
    st.subheader("Chat Statistics")
    st.metric("Messages", f"{st.session_state.message_count}/10")
    
    if st.session_state.message_count >= 10 and not st.session_state.memory_extracted:
        st.warning("⚠️ Ready to extract memory!")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.session_state.extracted_memory = None
        st.session_state.memory_extracted = False
        st.rerun()
    
    if st.session_state.message_count >= 10:
        if st.button("🧠 Extract Memory", use_container_width=True, type="primary"):
            with st.spinner("Analyzing conversation..."):
                memory_data = st.session_state.memory_extractor.extract_memory(
                    st.session_state.chat_history
                )
                st.session_state.extracted_memory = memory_data
                st.session_state.memory_extracted = True
                st.success("Memory extracted!")
                st.rerun()

st.markdown('<div class="main-header">🤖 Companion AI System</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memory Extraction", "🎭 Personality Demo"])

with tab1:
    st.subheader(f"Chat Interface - Current Personality: {st.session_state.current_personality}")
    
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "personality" in message:
                    st.caption(f"🎭 Personality: {message['personality']}")
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.message_count += 1
        
        with st.spinner("Thinking..."):
            response = st.session_state.personality_engine.generate_response(
                prompt,
                st.session_state.chat_history[:-1],  
                st.session_state.current_personality
            )
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "personality": st.session_state.current_personality,
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()

with tab2:
    st.header("Memory Extraction Results")
    
    if st.session_state.extracted_memory:
        memory = st.session_state.extracted_memory
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 User Preferences")
            if memory.get("user_preferences"):
                for pref in memory["user_preferences"]:
                    st.markdown(f"- {pref}")
            else:
                st.info("No preferences detected yet")
        
        with col2:
            st.subheader("😊 Emotional Patterns")
            if memory.get("emotional_patterns"):
                for pattern in memory["emotional_patterns"]:
                    st.markdown(f"- {pattern}")
            else:
                st.info("No patterns detected yet")
        
        with col3:
            st.subheader("📌 Key Facts")
            if memory.get("facts_to_remember"):
                for fact in memory["facts_to_remember"]:
                    st.markdown(f"- {fact}")
            else:
                st.info("No facts recorded yet")
        
        st.divider()
        
        st.subheader("📄 Complete JSON Output")
        json_str = json.dumps(memory, indent=2)
        st.code(json_str, language="json")
        
        st.download_button(
            label="📥 Download Memory JSON",
            data=json_str,
            file_name=f"memory_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("💡 Chat for at least 10 messages, then click 'Extract Memory' in the sidebar to analyze the conversation.")
        
        if st.session_state.message_count > 0:
            st.progress(st.session_state.message_count / 10)
            st.caption(f"{st.session_state.message_count}/10 messages")

with tab3:
    st.header("Personality Engine Demonstration")
    st.write("See how the same message changes with different personalities!")
    
    demo_message = st.text_area(
        "Enter a sample user message:",
        value="I've been feeling stressed about my upcoming exams and don't know how to manage my time.",
        height=100
    )
    
    if st.button("Generate Responses", type="primary"):
        st.subheader("Personality Comparison")
        
        demo_personalities = [
            "Neutral Assistant",
            "Calm Mentor", 
            "Witty Friend",
            "Therapist"
        ]
        
        with st.spinner("Generating responses..."):
            for personality in demo_personalities:
                with st.expander(f"🎭 {personality}", expanded=True):
                    response = st.session_state.personality_engine.generate_response(
                        demo_message,
                        [],
                        personality
                    )
                    st.write(response)
                    st.divider()

st.sidebar.divider()
st.sidebar.caption(f"Model: {st.session_state.current_model}")
st.sidebar.caption(f"Personality: {st.session_state.current_personality}")
