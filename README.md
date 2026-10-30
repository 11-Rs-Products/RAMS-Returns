# RAMS-Returns

## SQLite Database Setup

This project is set up with an SQLite database configuration supporting both standard library `sqlite3` and optional `SQLAlchemy`.

### Setup Instructions

1. **Install Dependencies (Optional)**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Python includes `sqlite3` in the standard library, so no extra dependencies are strictly required to get started.)*

### Running the Streamlit Chatbot

1. **Configure Environment**:
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY=your_key
   ```

2. **Run the App**:
   ```bash
   python3 -m streamlit run app.py
   ```
   Or specify a port if needed:
   ```bash
   python3 -m streamlit run app.py --server.port 8501
   ```

3. **Features**:
   - ChatGPT-like streaming chat interface.
   - Separate LLM connection via `google-genai` in [`llm_client.py`](file:///Users/reyanshmanta/Code%20and%20all/RAMS-Returns/llm_client.py).
   - Gemini Flash model selection (default: `gemini-2.5-flash`).
   - Automatically stores conversation history in SQLite database ([`database.py`](file:///Users/reyanshmanta/Code%20and%20all/RAMS-Returns/database.py) & [`schema.sql`](file:///Users/reyanshmanta/Code%20and%20all/RAMS-Returns/schema.sql)).
   - **Student Task Manager**:
     - Natural language task creation, updates, completion, and deletion.
     - Dedicated `tasks` SQLite table with priority, status, and deadline constraints.
     - Deadline intelligence: identifies overdue tasks, tasks due today/this week, and remaining study effort.
     - Interactive Streamlit dashboard with real-time metrics, status filters, manual task editing, and safe deletion.
   - Long-term memory extraction, cross-session context injection, and audit log.
   - **Opportunity Discovery**:
     - Searches hackathons, student competitions, coding competitions, and innovation challenges using the Brave Search API when configured.
     - Saves source URLs, retrieval times, field-level confidence, and source history in SQLite.
     - Uses `BRAVE_SEARCH_API_KEY` in `.env` for live discovery. Without it, RAMS reports that search is unavailable rather than fabricating results.
     - Discovery is restricted to individual Unstop and Devpost hackathon listings; platform filters are available in the Opportunities tab.
   - **NVIDIA Nemotron** (optional): set `RAMS_LLM_PROVIDER=nvidia`, `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and `NVIDIA_MODEL`. The default verified NVIDIA endpoint is `https://integrate.api.nvidia.com/v1` with `nvidia/nemotron-3-nano-30b-a3b`; RAMS uses an OpenAI-compatible chat request and routes returned tool calls through the existing registry.
   - **OpenShell** (optional): RAMS includes a restricted wrapper in [`openshell_runner.py`](</Users/reyanshmanta/Code%20and%20all/RAMS-Returns/openshell_runner.py>) for an operator-created sandbox. RAMS does not claim to be sandboxed unless launched inside that sandbox. Verify availability with `openshell --version`.

### Running the Test Suites

1. **Student Task Manager Test Suite**:
   ```bash
   python3 test_student_tasks.py
   ```
2. **Memory Pipeline Test Suite**:
   ```bash
   python3 test_memory_extraction.py
   ```
3. **Opportunity Discovery Test Suite**:
   ```bash
   python3 test_opportunity_service.py
   ```
4. **NVIDIA/OpenShell integration tests**:
   ```bash
   python3 test_nvidia_integration.py
   ```
