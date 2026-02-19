import streamlit as st
import os
import sys
import re
import logging

# 1. SET ENVIRONMENT VARIABLES FIRST ---
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" 
os.environ["HF_TOKEN"] = ""  # This kills the bad character error
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true" # Disable CrewAI telemetry to stop the signal handler error
os.environ["OTEL_SDK_DISABLED"] = "true" # Disable CrewAI telemetry to stop the signal handler error
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import shutil

# 2. CORE IMPORTS
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 2. COMMUNITY & PARTNER PACKAGES
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.memory import ConversationSummaryMemory

# 3. CREWAI & COMPATIBILITY HELPERS
from crewai import Agent, Task, Crew, Process
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tools import BaseTool
from crewai import LLM
from pydantic import Field
from pydantic import BaseModel
from typing import Any, Union, Dict
import json

import warnings
from langchain_core._api import LangChainDeprecationWarning

# This silences the specific "Chroma" and "LangChain" nagging warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

class CompanySummary(BaseModel):
    summary: str

# --- 4. UI CONFIGURATION ---
st.set_page_config(page_title="EngagePro AI Engineer", page_icon="⚙️")
st.title("⚙️ EngagePro AI Engineer")

# SETUP THE PDF TOOL
# Force the tool to use local embeddings so it stops calling OpenAI
@st.cache_resource
def get_vectorstore():
    # Use a versioned directory name to avoid lock conflicts
    # If the folder is locked, just create 'chroma_db_v2'
    persist_dir = "./chroma_db_v2" 

    # Force the model to load on CPU explicitly
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
           
    if not os.path.exists(persist_dir):
        # 1. Create a placeholder for the status message
        status_msg = st.empty()
        status_msg.info("📦 Creating new search index...")
        
        loader = PyPDFLoader("Company_Brochure.pdf", extract_images=False) # Ensure it's not trying to OCR images if not needed
        data = loader.load()

        # 2. Minimal cleaning to remove layout junk but keep company info
        for doc in data:
            # Removes common footer junk and table symbols that break semantic search
            doc.page_content = doc.page_content.replace("© 2024 EngagePro. All rights reserved.", "")
            doc.page_content = doc.page_content.replace(',"', '').replace('"', '')

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""])
        splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
        
        # 3. DISAPPEAR the message when finished
        status_msg.empty()
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
    return vectorstore

# Global instances
vectorstore = get_vectorstore()
api_wrapper = WikipediaAPIWrapper()
wiki_engine = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- 5. TOOL DEFINITIONS ---
# --- 5.1. THE UNIVERSAL WRAPPER (Prevents Pydantic/Chroma Error) ---
class MyToolSchema(BaseModel):
    """Input schema for the Internal Company Search tool."""
    query: Union[str, Dict[str, Any]] = Field(
        ..., 
        description="The search query or properties dictionary containing the query."
    )

class SimpleCrewTool(BaseTool):
    name: str
    description: str
    lc_tool: Any = Field(exclude=True) 
    result_as_answer: bool = True

    # Tells CrewAI exactly how to validate inputs ---
    args_schema: Any = MyToolSchema

    def _run(self, query: Union[str, dict]) -> str:
        
        # Robust handling for different JSON formats from the Agent
        if isinstance(query, dict):
            # Check for standard 'query' OR the nested 'properties' found in your log
            search_query = query.get("query") or query.get("properties", {}).get("query")
            
            # Fallback if the agent sends a different structure
            if not search_query:
                search_query = next(iter(query.values()), str(query))
        else:
            search_query = query
            
        return self.lc_tool.run(search_query)

# --- 5.2. THE REFINED TOOL ---

@tool("Internal_Company_Search")
def company_search_tool(question: str) -> str:
    """
    Search the company brochure for all facts.
    """
    try:
        # Use the global 'vectorstore' object
        docs = vectorstore.similarity_search(
            query=question, 
            k=3, 
        )
        
        if not docs:
            return "DATA_NOT_FOUND: No relevant information in the brochure."

        # ROBUSTNESS FIX: Check if the query keywords actually exist in the chunks
        # This prevents generic 'we empower teams' text from blocking Wiki
        results = []
        found_actual_match = False
        # Filter out short common words to find meaningful keywords
        keywords = [w.lower() for w in question.split() if len(w) >= 2]

        for d in docs:
            content = d.page_content.lower()
            # If the chunk contains one of our keywords, it's a valid match
            if any(k in content for k in keywords):
                found_actual_match = True
            
            page = d.metadata.get('page', 0) + 1
            results.append(f"[Source: Page {page}]: {d.page_content}")

        # If none of the 5 chunks actually contained the keyword, trigger Wiki
        if not found_actual_match:
            return "DATA_NOT_FOUND: Subject not mentioned in EngagePro brochure."

        return "\n\n".join(results)
    except Exception as e:
        return f"Retrieval Error: {e}"

@tool("SearchPeopleAndTeam")
def people_search_func(query: str):
    """Search for founders, CEO Thomas Gay, and personnel."""
    # 1. Search the PDF
    result = company_search_tool.run(f"Team, leadership, CEO Thomas Gay: {query}")
    
    # 2. If PDF is empty or says not found, return the fallback trigger
    if not result or "DATA_NOT_FOUND" in result:
        return "DATA_NOT_FOUND: The internal brochure does not mention the CEO. Search external sources."
    
    return result

@tool("WikipediaSearch")
def wiki_func(query: str):
    """Search Wikipedia for general knowledge."""
    return wiki_engine.run(query)

# Wrap tools for CrewAI
company_search_tool_wrapped = SimpleCrewTool(
    name="Internal_Company_Search", 
    description="Search EngagePro products, mission, and internal facts. ONLY use for company-related data.", 
    lc_tool=company_search_tool, 
    result_as_answer=True
    )
people_search_tool = SimpleCrewTool(
    name="SearchPeopleAndTeam", 
    description="Find info on CEO Thomas Gay or staff. Use only for professional identity verification.", 
    lc_tool=people_search_func
    )
wiki_tool = SimpleCrewTool(
    name="WikipediaSearch", 
    description="""Search Wikipedia for TECHNICAL or GENERAL definitions (e.g., 'What is AI?'). 
    STRICTLY PROHIBITED: Do not search for politics, religion, or controversial social issues. 
    Violating this will trigger a compliance failure.""", 
    lc_tool=wiki_func
    )


# --- 6. LLM SETUP ---
llm = LLM(
    model="openai/meta-llama-3.1-8b-instruct",         # The 'openai/' prefix is required for local compatibility
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.4, # Increased slightly to prevent repetitive phrasing
    # Force LiteLLM to use the standard completion format
    extra_params={"stream": False}
)
# This must be in session_state to persist during chat reruns
# --- 6.1 MEMORY INITIALIZATION ---
# Initialize chat history ONCE
if "chat_messages_obj" not in st.session_state:
    st.session_state.chat_messages_obj = StreamlitChatMessageHistory(key="chat_messages")

msgs = st.session_state.chat_messages_obj

# --- 7. YOUR PRESERVED PROMPT ENGINEERING ---
template = """You are the Senior AI Engineer at EngagePro. You are professional and insightful.
Answer the following questions accurately using the tools provided.

## TOOLS AVAILABLE:
{tools}

## RULES:
1. ALWAYS include the source: (Source: Page X) or (Source: Wiki).
2. If the internal search fails, use Wikipedia.

Chat History:
{chat_history}

Question: {input}
Thought: I should always think about what to do. 
Action: the action to take, must be one of [{tool_names}]
Action Input: the specific input for the action
Thought: {agent_scratchpad}"""

prompt_template = PromptTemplate.from_template(template)

# --- 8. AGENT & CREW CONSTRUCTION ---

# Specialist 0: The Context Filter (GHOST & HALLUCINATION KILLER)
librarian_agent = Agent(
    role='EngagePro Context Librarian',
    goal='Identify established facts from chat history and filter out technical JSON noise.',
    backstory=(
        "You are a data sanitation expert. "  
        "You review previous chat turns to find what facts the user already knows about EngagePro. "
        "You strictly ignore JSON brackets {}. "
        "You only output a clean, human-readable bulleted list. "
        "If there is no history, say 'No prior context'. "
        "You review chat history and internal facts to brief the other agents."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Specialist 1: The Internal Searcher (THE RETRIEVER)
extraction_agent = Agent(
    role='EngagePro Data Retriever',
    goal='Search the brochure for {user_query}. Extract raw, relevant facts including financial performance, growth metrics, and personnel details.',
    backstory=(
        "You are a professional researcher. Your primary source of truth is the PDF brochure, "
        "augmented by the verified internal updates provided below. "
        
        "VERIFIED REVENUE UPDATES: "
        "While the PDF contains older records (2019-2020), the current verified figures are: "
        "2024 revenue is $118M and 2025 (last fiscal year) revenue is $242M. "
        "Only surface these updated figures if the query is about current revenue or the last fiscal year. "

        "INTERNAL BIOGRAPHICAL KNOWLEDGE: "
        "EngagePro was founded by CEO Thomas Gay (Tom Gay), a serial entrepreneur and software "
        "industry veteran (founder of Refer.com and NDS). Use this to identify relevant brochure sections. "
        
        "STRICT OPERATIONAL RULES: "
        "1. Never describe yourself or your training. "
        "2. Do not add personal opinions. "
        "3. If the query asks 'What is EngagePro?', report the mission statement exactly from the PDF. "
        "4. Do NOT mention Thomas Gay or revenue unless the query specifically asks for them."
    ),
    tools=[company_search_tool_wrapped],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Specialist 2: The Response Researcher
formatter_agent = Agent(
        role='EngagePro Response Architect',
        goal='Format raw text into a professional 2-sentence response.',
        backstory=(
        "You are a master of corporate communication and data integrity. "
        "Your role is to review the research data from the Retriever (PDF) and the Wiki Agent. "
        "HIERARCHY OF TRUTH: "
        "1. Always prioritize the brochure/PDF data. If the PDF provides a fact, ignore Wikipedia. "
        "2. Only use Wikipedia data if the PDF Retriever states it could not find the information. "
        "3. Ensure the tone is professional, and the response is strictly 2 sentences long."
        "4. You are a minimalist editor. You hate long words and extra sentences."
        "5. Your job is to take technical."
        "6. You never use tools; you only process text. You only report facts found in the search."
    ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Specialist 3:  The External Researcher (Wiki)
wiki_agent = Agent(
    role='EngagePro Technical Specialist',
    goal='Explain technical AI concepts using Wikipedia to supplement company information.',
    backstory=(
        "You are an expert in AI. If you search for 'CX' or 'Transformer', "
        "STRICTLY IGNORE results related to Mazda automobiles, electrical hardware, "
        "or the 'Transformers' movie franchise. Focus only on Customer Experience (CX) "
        "and AI Transformer models."
    ),
    tools=[wiki_tool], 
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Specialist 4: The Compliance Checker
compliance_agent = Agent(
    role="Corporate Compliance Officer",
    goal="Ensure all responses are professional and factually grounded without adding extra data.",
    backstory="""You are a compliance officer at EngagePro, Singapore. 
    Your job is to block politics, religion, or unprofessional tone. 
    As long as the facts provided by the architect match the brochure (like the mission, 
    Singapore location, or revenue), the response is SAFE. 
    Do NOT refuse a response just because it is missing revenue or CEO details; 
    only refuse if it contains BANNED topics like politics or religion.""",
    llm=llm, 
    allow_delegation=False,
    verbose=True
)

def input_guardrail(query: str):
    """
    Validates and cleans user input before it reaches the AI agents.
    Prevents Singapore-policy violations AND Prompt Injections.
    Returns (is_safe, processed_query, error_message)
    """
    query_lower = query.lower()

    # --- A. INJECTION & JAILBREAK DETECTOR ---
    # Patterns used to override system instructions
    jailbreak_patterns = [
        "ignore all previous", "disregard all instructions", 
        "system prompt", "you are now a", "act as a", 
        "developer mode", "jailbreak", "bypass", "hidden instructions",
        "output your initial", "what is your prompt"
    ]

    if any(pattern in query_lower for pattern in jailbreak_patterns):
        return False, None, "Access Denied: Attempt to override system policy detected."
    
    # --- B. POLITICS & RELIGION FILTER (Singapore Context) ---
    # Keywords that trigger a block based on company policy
    banned_topics = [
        "politics", "pap", "pwp", "election", "government policy",
        "religion", "god", "church", "mosque", "temple", "halal", "haram",
        "protest", "strike", "activism"
    ]
    
    if any(topic in query.lower() for topic in banned_topics):
        return False, None, "I'm sorry, but EngagePro policy restricts discussions regarding politics or religion to ensure a professional workplace. Please keep questions focused on our products and operations."

    # --- C. PII MASKING (Singapore Specific) ---
    # 1. Singapore NRIC/FIN Pattern (S/T/F/G followed by 7 digits and a letter)
    nric_pattern = r"[STFG]\d{7}[A-Z]"
    # 2. Singapore Mobile Number (Starts with 8 or 9, 8 digits total)
    sg_phone_pattern = r"(?:\+65\s?)?[89]\d{7}"

    safe_query = re.sub(nric_pattern, "[MASKED_NRIC]", query, flags=re.IGNORECASE)
    safe_query = re.sub(sg_phone_pattern, "[MASKED_PHONE]", safe_query)

    return True, safe_query, None

def run_crew_logic(input_data, librarian_agent, extraction_agent, formatter_agent, wiki_agent):
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    user_query = input_data.get("user_query", "")
    q_lower = user_query.lower().strip()

    # --- STEP 1: SHORT-CIRCUIT LOGIC FOR CEO & REVENUE ---
    # Catch CEO/Founder requests
    if any(word in q_lower for word in ["ceo", "founder", "tom gay"]):
        return 'EngagePro was founded by CEO Thomas Gay (Tom Gay), a serial entrepreneur and software industry veteran.'

    # Catch ALL variations of revenue/money/financial growth
    # This prevents the agents from ever waking up for these questions
    if "revenue" in q_lower:
        if not any(word in q_lower for word in ["previous", "past", "2019", "2020", "history"]):
             return r"EngagePro reported a verified revenue of \$118M in 2024 and \$242M in the last fiscal year (2025)."
    
    # --- STEP 2: DATA PREPARATION ---
    raw_history = [f"{m.type}: {m.content}" for m in msgs.messages]
    history_str = "\n".join(raw_history) if raw_history else "No history yet."
    
    # --- STEP 3: DEFINE DEFINE CONSTANTS & SCENARIOS ---
    # These scenarios define how the agents will search the PDF.
    
    # Define the constants: technology terms triggers found in the brochure
    tech_keywords = [
    # Natural Language Suite
    "nlp", "natural language processing", 
    "nlu", "natural language understanding", 
    "nlg", "natural language generation",
    
    # AI & Machine Learning Core
    "generative ai", "gen ai", "ai-driven", "machine learning", "ml",
    "llm", "large language models", "transformer", "agentic", "predictive analytics",
    
    # EngagePro Specific / Product Terms
    "sentiment analysis", "real-time analytics", "omnichannel", "cx", "customer experience", "crm",
    
    # Advanced / Emerging Concepts
    "multi-turn", "conversational ui", "xai", "explainable ai", "explainable artificial intelligence"
    ]
    
    # --- STEP 4: SCENARIO ROUTING (The if/elif chain) ---

    # This chain now only handles things that WEREN'T caught by the short-circuit above.
    
    # Dedicated CX Transformer Search
    if any(prod in q_lower for prod in ["cx transformer", "cx customer", "cx"]):
        # SCENARIO 1: Dedicated CX Transformer Search
        target_desc = (
            "The user is asking about the CX Transformer. "
            "Search the brochure specifically in the 'Overview' or 'Product Suite' sections. "
            "Locate the performance data regarding the 40% reduction in resolution times. "
            "If the tool returns DATA_NOT_FOUND for 'CX', search for 'Customer Experience'."
        )
        target_expect = "Technical specifications and performance metrics for CX Transformer from the Overview section."

    # Dedicated InnovaBot Search
    elif "innovabot" in q_lower:
        # SCENARIO 2: Dedicated InnovaBot Search
        target_desc = (
            "The user is asking about InnovaBot. "
            "Search the brochure for 'InnovaBot' and identify its primary use case "
            "for internal knowledge management among Fortune 500 companies. "
            "Extract any specific technical features listed for this product."
        )
        target_expect = "Specific technical and performance facts about InnovaBot retrieved from the brochure."
    
    # General Company Identity 
    elif any(word in q_lower for word in ["who is engagepro", "about engagepro", "company info"]) or q_lower == "what is engagepro":
        # SCENARIO 3: Company Identity Specific Instruction
        target_desc = "Search Page 1 specifically for 'Company Brief' and 'Company Vision'."
        target_expect = "A 2-sentence summary based on the 'Company Brief' found on Page 1 of the brochure."

    # Mission & Vision Specifics
    elif any(word in q_lower for word in ["mission", "vision", "goal", "aspire"]):
        # SCENARIO 4: Mission & Vision Specifics
        target_desc = "Search Page 1 for the exact 'COMPANY VISION' and 'Company Brief' paragraphs."
        target_expect = "The official mission and vision statement of EngagePro."

    # Add "headquarters" or "location" 
    elif any(word in q_lower for word in ["headquarter", "location", "address", "office"]):
        # SCENARIO 5: Location Specific Instructions
        target_desc = (
            "Search the brochure for the company headquarter or primary office location. "
            "Look for a Singapore-based address. "
        )
        target_expect = "The official physical address of EngagePro from the brochure."
   
    elif any(word in q_lower for word in ["revenue", "financial", "growth"]):
        # This block is skipped for "What is EngagePro?"
        # Now 'previous' triggers the specific revenue search instructions
        # SCENARIO 6: Financial/Revenue Query
        # CHECK IF USER WANTS HISTORICAL DATA
        if any(word in q_lower for word in ["previous", "past", "2019", "2020"]):
            target_desc = "Search the PDF specifically for '$35M' or '$50M' and the years '2019' or '2020'."
            target_expect = "The historical figures from the brochure: $35M in 2019 and $50M in 2020."
        else:
            target_desc = "Provide the latest verified revenue figures."
            target_expect = "$118M for 2024 and $242M for 2025."

    elif any(word in q_lower for word in ["achievement", "accomplishment", "milestone", "success"]):
        # SCENARIO 7: Achievements/Accomplishments - Target the specific bullet points on Page 3
        target_desc = (
            "The user wants SPECIFIC DATA POINTS from the 'Key Achievements' section. "
            "Search specifically for these three items: "
            "1. 'InnovaBot' (adopted by Fortune 500), "
            "2. 'CX Transformer' (40% reduction), "
            "3. '2019' (Top 10 AI startup). "
            "Do NOT return general marketing mission statements."
        )
        target_expect = "The 3 specific factual milestones listed in the brochure."
    
    elif any(word in q_lower for word in ["contact", "phone", "email", "website", "call"]):
        # SCENARIO 8: Contact-Specific 
        target_desc = (
            "Search the brochure for 'Contact Information'. "
            "You MUST extract all three: "
            "1. Email: info@engagepro2AI.com "
            "2. Phone: +65 9966 3500 2 "
            "3. Website: www.engagepro2AI.com"
        )
        target_expect = "A list containing the email, phone number, and website URL."

    elif any(word in q_lower for word in ["social", "linkedin", "twitter", "facebook", "handle", "follow"]):
        # SCENARIO 9: Social Media Handles
        target_desc = (
            "The user wants the official Social Media handles. "
            "Ignore descriptions of 'omnichannel support'. "
            "Search specifically for LinkedIn and Twitter/X handles. "
            "Look for '@EngagePro2AI' or specific platform names."
        )
        target_expect = "The specific social media handles (e.g., LinkedIn: EngagePro, Twitter: @EngagePro2AI)."

    elif any(word in q_lower for word in tech_keywords):
        # SCENARIO 10: Hybrid Search for technical terminology, Routes to both the PDF and Wikipedia
        target_desc = (
            f"The user is asking about a technical concept or product: '{input_data['user_query']}'.\n"
            "1. Extraction Agent: Search the EngagePro brochure FIRST. If 'CX' or 'Transformer' is mentioned, "
            "it refers to the 'CX Transformer' platform, not cars.\n"
            "2. Wiki Agent: ONLY provide a technical definition of the underlying technology (e.g., what is a 'Transformer' in AI? or 'What is Customer Experience?' or 'What is NLP?').\n"
            "3. Formatter Agent: If both provide data, lead with the EngagePro product info."
        )
        target_expect = "A professional response linking EngagePro's products to their technical definitions."

    else:
        # This block now only catches miscellaneous queries not covered by
        # Company Idenity, Mission, Location, Revenue, Achievement, Products, CEO, Contact-Specifics or Social Media- Speciifcs.
        # Scenario 11: Other User Queries 
        target_desc = (
            f"The user is asking a general question: '{user_query}'. "
            "Search the internal brochure for any remaining factual matches "
            "not covered by the core categories (Location, Products, Revenue)."
        )
        target_expect = "A factual and direct answer extracted from the brochure text."

    # Step 5: DEFINE AGENT TASKS (Using the variables above) ---
    # 8.1 Librarian Task (Clean History)
    librarian_task = Task(
    description=(
        "Review the chat history: {chat_history}. "
        "Extract only facts mentioned in the conversation about EngagePro. "
        "STRICT RULE: Do NOT describe your role as a librarian or data cleaner. "
        "If there is no history, return: 'No previous facts established.'"
    ),
    expected_output="A bulleted list of facts extracted from the conversation history.",
    agent=librarian_agent
    )

    # 8.2 Retiriever_task
    retriever_task = Task(
        description=target_desc,
        expected_output=target_expect,
        agent=extraction_agent,
        context=[librarian_task]
    )

    # 8.2 Conditional Wiki Task (The Router)
    # This function decides if it should search Wikipedia
    def check_if_wiki_needed(output):
        # 1. Standard check for empty results
        if not output.raw or "DATA_NOT_FOUND" in output.raw:
            return True
            
        # 2. CEO BYPASS: If the user is asking for CEO/Founder, DO NOT go to Wikipedia.
        # Use the internal knowledge from the agent backstory.
        q_lower = user_query.lower()
        ans_lower = output.raw.lower()

        # Define internal topics that should NEVER trigger Wiki
        internal_topics = ["thomas gay", "ceo", "revenue", "innovabot", "cx transformer", "engagepro", "founder", "thomas gay"]
        
        # If the query is NOT internal, but the answer doesn't mention the query subject:
        if not any(topic in q_lower for topic in internal_topics):
            subject = q_lower.split()[-1] # Simple way to get the main subject
            if subject not in ans_lower:
                return True # The brochure returned filler; go to Wikipedia
            
        return False

    wiki_task = ConditionalTask(
        description=(
        f"The internal archives do not contain specific data on: {user_query}. "
        "Acting as a Technical Consultant, search Wikipedia to provide a "
        "comprehensive general definition or technical overview of this topic."
    ),
    expected_output="A professional technical summary from external records.",
    agent=wiki_agent,
    condition=check_if_wiki_needed,
    context=[retriever_task, librarian_task]
    )     
    
    # 8.3  Formatter Task (Response Architect)
    formatter_task = Task(
    description=(
        "Synthesize the research findings into a professional 2-sentence response. "
        "STRICT RULE: Your response must contain ONLY the information requested in the CURRENT query: {user_query}. "

        # 1. THE GHOST-KILLER (Absolute Override)
        "IF the query is exactly 'What is EngagePro?', output the following paragraph ONLY and STOP: "
        "'EngagePro is a leading technology firm committed to transforming how businesses interact with customers "
        "and maximize productivity. Specializing in generative artificial intelligence (AI), EngagePro develops "
        "innovative applications that streamline operations, boost customer satisfaction, and empower workforces.' "

        # 2. MISSION STATEMENT LOGIC
        "IF the query is about the 'Mission' or 'Vision', state: 'At EngagePro, we aspire to create a world where "
        "businesses can seamlessly connect with their customers, delivering personalized experiences.' "
        "FORBIDDEN: Do not mention revenue or products in a Mission/Vision answer. "

        # 3. REVENUE (Isolated with Conditional Logic)
        "IF the Retriever found historical data (2019/2020), report those figures. "
        "IF the Retriever found modern data (2024/2025), report those figures. "
        "Always match the timeframe requested by the user: {user_query}."
        "FORBIDDEN: Do not mention Singapore, the CEO, or mission statements in a revenue answer. "

        # 4. CEO (Isolated)
        "IF the query is about the CEO or Founder, state ONLY: 'EngagePro was founded by CEO Thomas Gay (Tom Gay), a serial entrepreneur and software industry veteran.' "
        "FORBIDDEN: Do not mention revenue or location in a CEO answer. "

        # 5. CONTACT & SOCIALS (Isolated)
        "IF query is for contact/website: Provide Email, Phone, and Website ONLY. "
        "IF query is for social media: Provide Facebook, Twitter, and LinkedIn handles ONLY. "

        # 6. LOCATION & STAFF (Isolated)
        "IF query is for location: State ONLY the Singapore International Business Park address. "
        "IF query is for staff: State ONLY the count of 500+ professionals. "

        # 7. GLOBAL NEGATIVE CONSTRAINTS
        "STRICT NEGATIVE CONSTRAINT: Never combine revenue, CEO, and location in the same response unless the user asked for multiple topics. "
        "If a fact is not a direct answer to {user_query}, DELETE IT from your final output."
    ),
    expected_output="A single 2-sentence response containing ONLY the direct answer to {user_query}. No extra facts.",
    agent=formatter_agent,
    context=[retriever_task, wiki_task] 
    )   

    # 8.4 Compliance Task (Final Verification)
    compliance_task = Task(
        description="""Review the response for corporate safety. 
        Ensure it answers the user query: {user_query} professionally. 
        If the response is a mission statement, location, or revenue figure found in the brochure, 
        approve it. DO NOT add any new information or introductory phrases.""",
        expected_output="The exact 2-sentence string provided by the formatter, with no additional commentary.",
        agent=compliance_agent,
        context=[formatter_task]
    )

    # --- 9 Execute Internal Search ---
    with st.status("🔍 Consulting EngagePro Archives...") as status:
        internal_crew = Crew(
            agents=[librarian_agent, 
                    extraction_agent, 
                    wiki_agent,
                    formatter_agent, 
                    compliance_agent],
            tasks=[librarian_task, 
                   retriever_task, 
                   wiki_task, 
                   formatter_task, 
                   compliance_task],
            process=Process.sequential,
            verbose=False
        )
        
        # Consistent variable name: final_response
        final_response = internal_crew.kickoff(inputs={
            "user_query": user_query,
            "chat_history": history_str
        })

        # ---9.1 Safe Extraction Logic  ---
        answer_text = "" # Initialize as empty string first

        # 9.2 Try Pydantic first
        if hasattr(final_response, 'pydantic') and final_response.pydantic:
            answer_text = final_response.pydantic.summary

        # 9.3 Simplified String Extraction (Only if 3.1 didn't fill it)
        if not answer_text:
            raw_output = str(final_response.raw)
                
            # Strip any accidental JSON/Markdown markers just in case
            if "```" in raw_output:
                raw_output = raw_output.split("```")[-2] if len(raw_output.split("```")) > 1 else raw_output
        
            if '"summary":' in raw_output:
                answer_text = raw_output.split('"summary":')[-1].strip(' "{}')
            else:
                answer_text = raw_output.strip()
                

        # 9.4 Accurate Status Feedback 
        # If the wiki_task was SKIPPED, it means the brochure info was enough
        if wiki_task.output is None:
            status.update(label="✅ Answered from EngagePro Brochure.", state="complete")
        else:
            status.update(label="✅ Answered with help from public records.", state="complete")

    # --- 10 Final Output and Memory Update ---
    # Logic to update your session summary memory without crashing
    # Use the actual memory object stored in session_state, not the history string.
    try:
        if "summary_memory" in st.session_state:
            st.session_state.summary_memory.save_context(
                {"input": user_query}, 
                {"output": answer_text}
            )
    except Exception as e:
        # Fallback to a simple string if the summary logic fails
        st.session_state['summary_memory_fallback'] = f"The user learned: {answer_text}"

    # This ensures any '$' in the text is escaped so Streamlit doesn't treat it as LaTeX
    # r"\$" uses a 'raw string' to avoid the SyntaxWarning while fixing the UI
    clean_text = answer_text.replace("\\", "")
    return clean_text.replace("$", r"\$ ")

# --- 11 MAIN UI CHAT INTERFACE APPLICATION ---
def main():
    
    # 11.1 THE DISPLAY LOOP: (History)
    for msg in msgs.messages:
        role = "assistant" if msg.type == "ai" else "user"
        with st.chat_message(role):
            # Clean and then escape for history rendering
            display_text = msg.content.replace("\\", "").replace("$", r"\$ ")
            st.markdown(display_text)
            
    # 11.2 THE ACTION GATE
    if user_query := st.chat_input("Ask a question about EngagePro..."):
        # Display user input immediately for responsiveness
        st.chat_message("user").write(user_query)

        # --- APPLY GUARDRAIL ---
        is_safe, safe_query, error_msg = input_guardrail(user_query)
    
        if not is_safe:
            st.warning(error_msg)
            # Record the interaction but show the guardrail refusal
            msgs.add_user_message(user_query)
            msgs.add_ai_message(error_msg)
            # No rerun needed here as st.warning and add_ai_message handle the immediate UI
        else:
            try:
                # 11.2.1 HANDLE HISTORY (If empty, provide a clear string)
                if len(msgs.messages) == 0:
                    history_str = "No previous history."
                else:
                    # Take last 4 messages and format them
                    recent_messages = msgs.messages[-4:]
                    history_str = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])
        
                # 11.2.2 SAVE NEW MESSAGE (We save the raw query for the user to see)
                msgs.add_user_message(user_query)
        
                # 11.2.3 EXECUTE LOGIC
                # Note: Passing 'safe_query' to the crew for processing
                answer_text = run_crew_logic(
                    {
                        "user_query": safe_query,
                        "chat_history": history_str 
                    }, 
                    librarian_agent=librarian_agent,
                    extraction_agent=extraction_agent, 
                    formatter_agent=formatter_agent,
                    wiki_agent=wiki_agent
                )

                # 11.2.4 SAVE AI MESSAGE
                msgs.add_ai_message(answer_text)

                # THE GHOST-KILLER: Redraws the UI with the new history
                st.rerun()
                                                                
            except Exception as e:
                st.error(f"Technical hitch: {e}")
 
# --- 12. ENTRY POINT ---
if __name__ == "__main__":
    main()