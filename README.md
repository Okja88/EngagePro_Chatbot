# EngagePro_Chatbot
This is a Ngee Ann Polytechnic School Assignment creating a chatbot utilizing Open Source from LLMs, RAG, external API call to Wikipedia, Langchain, Chroma DB, Hugging Face and OpenAi.

This README provides an overview of the EngagePro AI Engineer application, a sophisticated RAG (Retrieval-Augmented Generation) system built with Streamlit, LangChain, and CrewAI.

⚙️ EngagePro AI EngineerEngagePro AI Engineer is a specialized multi-agent system designed to act as a Senior AI Engineer for EngagePro. It intelligently retrieves information from an internal company brochure (PDF) and supplements it with technical definitions from Wikipedia using a coordinated "crew" of AI agents.

🚀 Features:
1) Multi-Agent Orchestration: Utilizes CrewAI to manage specialized agents (Librarian, Retriever, Researcher, Architect, and Compliance).
2) Hybrid RAG: Combines local PDF vector search (via ChromaDB and HuggingFace Embeddings) with real-time Wikipedia lookups.
3) Strict Guardrails: Includes a custom Singapore-context-aware guardrail system to filter sensitive topics (politics, religion) and mask PII (NRIC, phone numbers).
4) Contextual Memory: Tracks conversation history using StreamlitChatMessageHistory to provide coherent, multi-turn dialogue.
5) Local LLM Support: Configured to interface with local inference servers (like LM Studio) using the OpenAI-compatible API.

🏗️ System Architecture
The application follows a sequential workflow where data passes through multiple specialized layers:
The Crew of Agents
1) Context Librarian: Cleans and summarizes chat history to prevent "hallucination" and noise.
2) Data Retriever: Specialized in searching the Company_Brochure.pdf for specific facts and financial metrics.
3) Technical Specialist (Wiki): A conditional agent that triggers only when internal data is insufficient, providing general AI definitions.
4) Response Architect: Synthesizes findings into a professional, strictly formatted 2-sentence response.
5) Compliance Officer: The final gatekeeper ensuring the output meets corporate safety standards.

🛠️ Technical Stack
Component-----------Technology
1) Frontend---------Streamlit
2) Orchestration----CrewAI
3) LLM Framework----LangChain / LiteLLM
4) Vector Database--ChromaDB
5) Embeddings-------HuggingFace (all-MiniLM-L6-v2)
6) Search Tools-----Wikipedia API

📋 PrerequisitesPython: 
1) 3.10+Local 
2) LLM Server: LM Studio (or similar) running a model (default: llama-3.1-8b-instruct) at http://localhost:1234/v1.
3) Required Files: A file named Company_Brochure.pdf must be present in the root directory.

📂 Project Structure
To run the application correctly, ensure your directory is organized as follows:

```text
.
├── .gitignore
├── main.py                # Main Streamlit application script
├── Company_Brochure.pdf   # Source document for the RAG system
├── README.md              # Project documentation
├── requirements.txt       # List of Python dependencies
└── chroma_db_v2/          # Vector database folder (auto-generated)

### Why this specific layout matters for your `chat.py`:

1.  **`Company_Brochure.pdf`**: Your code specifically looks for this filename in the same folder where you run the command. If it’s missing or named differently, the `PyPDFLoader` in your script will crash.
2.  **`chroma_db_v2/`**: Your script is programmed to check if this folder exists. 
    * If it **doesn't** exist, the script creates it by "reading" your PDF.
    * If it **does** exist, it saves time by loading the data directly from this folder instead of re-reading the PDF.
3.  **`requirements.txt`**: As we discussed, this stays in the "root" (main) folder so that when you run `pip install -r requirements.txt`, the terminal finds it immediately.

🔧 Installation & Setup
1) Clone the repository and install dependencies:
pip install streamlit langchain crewai langchain-openai langchain-huggingface chromadb pydantic
2) Configure Environment Variables:The script handles several environment variables internally (e.g., disabling telemetry), but ensure your local LLM server is active.
3) Run the Application: streamlit run chat.py

🛡️ Security & Compliance
The system implements an input_guardrail function that:
1) Blocks Prompt Injections: Detects "jailbreak" phrases designed to override system instructions.
2) Singapore Policy Alignment: Restricts discussion on sensitive topics like local politics or religion.
3) Data Masking: Automatically masks Singapore NRIC/FIN patterns and mobile numbers using regex:
i) NRIC: $[STFG]\d{7}[A-Z]$
ii) Phone: $(?:+65)?[89]\d{7}$

💡 Usage Examples
1) Company Info: "What is EngagePro's mission?"
2) Financials: "What was the revenue in 2025?" (System uses hardcoded verified updates for recent years).
3) Technical: "Explain what an AI Transformer is." (Triggers the Wiki Agent).Contact: "How can I contact the team?"
