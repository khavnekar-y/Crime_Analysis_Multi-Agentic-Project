import streamlit as st
import requests
import json
import toml

# -------------------------------
# 1) Page Config for Wide Layout
# -------------------------------
st.set_page_config(
    page_title="NVIDIA Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2) Configuration
# -------------------------------
config = toml.load("config.toml")
API_URL = config["fastapi_url"]
QUERY_URL = f"{API_URL}/research_report"

# -------------------------------
# 3) Helper Functions
# -------------------------------
def display_financial_data(data):
    """Display Snowflake financial metrics and chart."""
    if not data:
        st.info("No financial data available")
        return

    st.markdown("### NVIDIA Financial Metrics")
    if "chart" in data:
        st.image(
            f"data:image/png;base64,{data['chart']}",
            caption="NVIDIA Valuation Metrics",
            use_column_width=True
        )
    if "metrics" in data:
        st.markdown("#### Key Metrics")
        if isinstance(data["metrics"], list):
            st.dataframe(data["metrics"])
        else:
            st.write(data["metrics"])


def display_rag_results(data):
    """Display results from the RAG Agent."""
    if not data:
        st.info("No document analysis available")
        return

    st.markdown("### Document Analysis Results")
    st.markdown(data.get("result", "No results found"))

    if "sources" in data:
        with st.expander("📚 Source Documents"):
            for src in data["sources"]:
                st.markdown(f"- {src}")


def display_web_results(data):
    """Display results from the Web Search Agent."""
    if not data:
        st.info("No web search results available")
        return

    st.markdown("### Web Search Results")
    st.markdown(data)

# -------------------------------
# 4) Sidebar Configuration
# -------------------------------
st.sidebar.title("NVIDIA Research Assistant")
st.sidebar.markdown("### Search Configuration")

search_type = st.sidebar.radio(
    "Select Search Type",
    ["All Quarters", "Specific Quarter"],
    key="search_type"
)

# Keep user’s selected periods in session
if "selected_periods" not in st.session_state:
    st.session_state.selected_periods = ["2023q1"]

if search_type == "Specific Quarter":
    all_periods = [f"{y}q{q}" for y in range(2020, 2026) for q in range(1, 5)]
    # 1) Filter out "all" from the default to avoid the Streamlit error
    default_selected = [
        p for p in st.session_state.selected_periods
        if p in all_periods
    ]
    if not default_selected:
        default_selected = ["2023q1"]  # Some safe default

    selected_periods = st.sidebar.multiselect(
        "Select Period(s)",
        options=all_periods,
        default=default_selected,  # Use the filtered list
        key="period_select"
    )

    if not selected_periods:
        selected_periods = ["2023q1"]
    st.session_state.selected_periods = selected_periods

else:
    selected_periods = ["all"]
    st.session_state.selected_periods = selected_periods


st.sidebar.markdown("---")
st.sidebar.markdown("### Agent Configuration")

if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = ["RAG Agent"]

available_agents = ["RAG Agent", "Web Search Agent", "Snowflake Agent"]
selected_agents = st.sidebar.multiselect(
    "Select AI Agents (at least one required)",
    options=available_agents,
    default=st.session_state.selected_agents,
    key="agent_select_unique"
)

# Validate agent selection
if not selected_agents:
    st.sidebar.warning("⚠️ At least one agent is required")
    selected_agents = ["RAG Agent"]
st.session_state.selected_agents = selected_agents.copy()

if st.sidebar.button("Apply Agent Selection", type="primary", use_container_width=True, key="apply_agents_unique"):
    st.session_state.selected_agents = selected_agents.copy()
    st.sidebar.success("✅ Agent selection updated!")

# -------------------------------
# 5) Navigation
# -------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

home_btn = st.sidebar.button("Home", key="nav_Home", use_container_width=True)
report_btn = st.sidebar.button("Combined Report", key="nav_Report", use_container_width=True)
about_btn = st.sidebar.button("About", key="nav_About", use_container_width=True)

if home_btn:
    st.session_state.current_page = "Home"
    st.rerun()
elif report_btn:
    st.session_state.current_page = "Combined Report"
    st.rerun()
elif about_btn:
    st.session_state.current_page = "About"
    st.rerun()

page = st.session_state.current_page

# -------------------------------
# 6) Page Layout
# -------------------------------
if page == "Home":
    st.title("Welcome to the NVIDIA Multi-Agent Research Assistant")
    st.markdown("""
        This application integrates multiple agents to produce comprehensive research reports on NVIDIA:
        - **RAG Agent**: Retrieves historical quarterly reports from Pinecone.
        - **Web Search Agent**: Provides real-time insights via SerpAPI.
        - **Snowflake Agent**: Queries structured valuation metrics and displays charts.
    """)

elif page == "Combined Report":
    st.title("NVIDIA Research Assistant")
    st.subheader("💬 Research History")

    # Show chat history
    with st.container():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                periods_text = ", ".join(message.get("selected_periods", []))
                agents_text = ", ".join(message.get("agents", []))
                st.markdown(f"""
                    <div class='user-message'>
                        <div class='metadata'>📅 {periods_text}<br>🤖 Agents: {agents_text}</div>
                        <div>🔍 {message['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='assistant-message'>
                        <div class='metadata'>🤖 NVIDIA Research Assistant</div>
                        <div>{message['content']}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    with st.form("report_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input("Research Question", placeholder="What has driven NVIDIA's revenue growth?")
        with col2:
            submitted = st.form_submit_button("➤")

    if submitted and question:
        with st.spinner("🔄 Generating report..."):
            payload = {
                "question": question,
                "search_type": search_type,
                "selected_periods": selected_periods,
                "agents": selected_agents
            }
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Add user message
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "search_type": search_type,
                        "selected_periods": selected_periods,
                        "agents": selected_agents
                    })
                    # Add assistant message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data.get("final_report", ""),
                        "rag_output": data.get("rag_output"),
                        "snowflake_data": data.get("valuation_data"),
                        "web_output": data.get("web_output"),
                        "agents": selected_agents
                    })
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Show results in tabs
    if st.session_state.chat_history:
        latest = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "assistant"), None)
        if latest:
            st.markdown("---")
            st.subheader("📊 Detailed Results")

            # Tabs based on which agents are active
            tabs_to_show = ["Overview"]
            if "Snowflake Agent" in selected_agents:
                tabs_to_show.append("Financial Data")
            if "RAG Agent" in selected_agents:
                tabs_to_show.append("Document Analysis")
            if "Web Search Agent" in selected_agents:
                tabs_to_show.append("Web Results")

            tab_objects = st.tabs(tabs_to_show)

            for i, tab_name in enumerate(tabs_to_show):
                with tab_objects[i]:
                    if tab_name == "Overview":
                        st.markdown(latest["content"])
                    elif tab_name == "Financial Data":
                        display_financial_data(latest.get("snowflake_data"))
                    elif tab_name == "Document Analysis":
                        # If we have a summarized version from the agent node, show that:
                        rag_summary = latest.get("rag_summary", None)
                        if rag_summary:
                            st.markdown("#### Summarized Historical Insights")
                            st.write(rag_summary)
                        else:
                            # fallback to raw chunks if summary not available
                            display_rag_results(latest.get("rag_output"))
                    elif tab_name == "Web Results":
                        web_summary = latest.get("web_summary", None)
                        if web_summary and web_summary != "No web data to summarize.":
                            st.markdown("### Web Summary")
                            st.write(web_summary)
                        else:
                            # fallback to the raw chunk
                            display_web_results(latest.get("web_output"))

elif page == "About":
    st.title("About NVIDIA Research Assistant")
    st.markdown("""
        **NVIDIA Multi-Agent Research Assistant** integrates:
        - **RAG Agent**: Uses Pinecone with metadata filtering to retrieve historical reports.
        - **Web Search Agent**: Uses SerpAPI for real-time search.
        - **Snowflake Agent**: Connects to Snowflake for valuation measures and charts.
    """)

# -------------------------------
# 7) Custom CSS (UNCHANGED)
# -------------------------------
st.markdown("""
<style>
/* ---------------------------------- */
/* Dark background, Nvidia green accent */
/* ---------------------------------- */
body, .main, [data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    font-family: "Segoe UI", sans-serif;
}

/* ---------------------------------- */
/* Make links NVIDIA green, no underline */
/* ---------------------------------- */
a, a:visited {
    color: #76B900 !important; /* Nvidia green */
    text-decoration: none !important;
}
a:hover {
    color: #5c8d00 !important; 
    text-decoration: underline !important;
}

/* ---------------------------------- */
/* Headings in NVIDIA green */
/* ---------------------------------- */
h1, h2, h3, h4 {
    color: #76B900 !important; /* Nvidia Green */
}

/* ---------------------------------- */
/* Block container width */
/* ---------------------------------- */
.block-container {
    max-width: 1400px; /* Full width container */
}

/* ---------------------------------- */
/* Chat Bubbles styling */
/* ---------------------------------- */
.chat-container {
    margin-bottom: 30px;
    max-height: 55vh; /* adjustable height */
    overflow-y: auto;
    padding: 1em;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    background-color: #2b2b2b;
}
.user-message {
    background-color: #2E8B57; /* or #2196F3 if you prefer bluish */
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}
.assistant-message {
    background-color: #262730;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}
.metadata {
    font-size: 0.8em;
    color: #B0B0B0;
    margin-bottom: 5px;
}

/* ---------------------------------- */
/* Circular submit button (ChatGPT style) */
/* ---------------------------------- */
[data-testid="stFormSubmitButton"] button {
    border-radius: 50%;
    width: 50px;
    height: 50px;
    padding: 0;
    min-width: 0;
    font-size: 1.4em;
    font-weight: bold;
    background-color: #76B900;
    color: #fff;
    border: none;
    transition: background-color 0.3s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #5c8d00;
}

/* ---------------------------------- */
/* Tab styling to match dark theme */
/* ---------------------------------- */
div[data-testid="stTabs"] button {
    background-color: #333333;
    color: #fff;
    border: none;
    border-radius: 0;
    padding: 0.5rem 1rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #76B900 !important;
    color: #000 !important;
    font-weight: bold;
}

/* ---------------------------------- */
/* Datatable override for dark theme */
/* ---------------------------------- */
[data-testid="stDataFrame"] {
    background-color: #262730;
    color: #fff;
    border: none;
}
[data-testid="stDataFrame"] table {
    color: #fff;
}

/* ---------------------------------- */
/* Sidebar background & text colors */
/* ---------------------------------- */
[data-testid="stSidebar"] {
    background-color: #2b2b2b !important; /* Dark gray background */
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stButton {
    color: #fff !important; /* White text */
}
[data-testid="stSidebar"] .stRadio > div:hover {
    background-color: #333333 !important;
}

/* -------------------------------------- */
/* Sidebar nav buttons in NVIDIA green   */
/* -------------------------------------- */
[data-testid="stSidebar"] .stButton > button {
    background-color: #0a8006 !important; /* NVIDIA green */
    color: #fff !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #5c8d00 !important; /* darker green hover */
    color: #fff !important;
}

/* ---------------------------------- */
/* Agent Selection Submit Button */
/* ---------------------------------- */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #76B900 !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
    transition: background-color 0.3s ease !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #5c8d00 !important;
}

/* ---------------------------------- */
/* Navigation Buttons Layout */
/* ---------------------------------- */
[data-testid="column"] {
    padding: 0.25rem !important;
}

[data-testid="stButton"] button {
    width: 100% !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)
