from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langsmith import Client
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
import requests
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Set USER_AGENT for web requests
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# LangSmith configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT_SEARCH_ENGINE", "autosar-agent")

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm_model = ChatGroq(
    model="llama-3.1-8b-instant", 
    groq_api_key=groq_api_key,
    temperature=0
)

print("=" * 60)
print("AUTOSAR CAN Interface Agent")
print("=" * 60)

# Load AUTOSAR PDF
print("\n1. Loading AUTOSAR PDF document...")
try:
    pdf_url = "https://www.autosar.org/fileadmin/standards/R25-11/CP/AUTOSAR_CP_SWS_CANInterface.pdf"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    print(f"   Downloading from: {pdf_url}")
    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()
    
    pdf_path = "autosar_can_interface.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    print(f"   Saved to: {pdf_path}")
    
    # Load with PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages")
    
except Exception as e:
    print(f"   Error loading PDF: {e}")
    print("   Creating sample documents...")
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content="""SOFTWARE FILTERING IN AUTOSAR CAN INTERFACE
            Software filtering in AUTOSAR CanIf enables selective message processing based on configurable filter criteria.
            
            Key Features:
            - Filter by CAN Identifier (Standard 11-bit or Extended 29-bit)
            - Filter by data pattern matching
            - Configurable filter masks through ECU Configuration
            - Multiple filter configurations supported per hardware unit
            
            Purpose: Reduce ECU CPU load by processing only relevant CAN messages.
            Implementation: Filter masks defined in CanIf module configuration.""",
            metadata={"source": "AUTOSAR_SWS_CANInterface.pdf", "page": 45, "section": "Filtering"}
        ),
        Document(
            page_content="""CAN INTERFACE (CanIf) MODULE OVERVIEW
            The CanIf module provides hardware-independent interfaces between CAN drivers and upper communication layers.
            
            Primary Functions:
            1. Message filtering and validation
            2. Buffer management for Rx/Tx messages
            3. Hardware abstraction for CAN controllers
            4. Error detection and notification
            5. State management of CAN controllers
            
            Architecture Position: Part of Communication Services layer in BSW.""",
            metadata={"source": "AUTOSAR_SWS_CANInterface.pdf", "page": 23, "section": "Overview"}
        ),
        Document(
            page_content="""CLASSIC AUTOSAR LAYERED ARCHITECTURE
            
            LAYERS:
            1. APPLICATION LAYER (SW-C): Application software components
            2. RUNTIME ENVIRONMENT (RTE): Communication middleware between SW-C and BSW
            3. BASIC SOFTWARE (BSW): Standardized automotive software services
               - Communication Services: COM, CanIf, CanSm, PduR
               - System Services: EcuM, BswM, Dem, Det
               - Memory Services: NvM, MemIf
            4. MICROCONTROLLER ABSTRACTION LAYER (MCAL): Hardware-specific drivers
            
            CanIf resides in Communication Services sub-layer of BSW.""",
            metadata={"source": "AUTOSAR_Introduction.pdf", "page": 5, "section": "Architecture"}
        ),
        Document(
            page_content="""FILTER MASK CONFIGURATION PARAMETERS
            
            Configuration Elements:
            - CanIfFilterMaskId: Unique identifier for filter mask
            - CanIfFilterMaskType: Filter type (IDENTIFIER_MASK, DATA_MASK, IDENTIFIER_DATA_MASK)
            - CanIfFilterMaskValue: Filter value to match against messages
            - CanIfFilterMaskRef: Reference to associated CAN controller
            
            Filtering Modes:
            - ACCEPTANCE: Messages matching filter are accepted
            - REJECTION: Messages matching filter are rejected
            
            Configuration Method: Through AUTOSAR ECU Configuration parameters.""",
            metadata={"source": "AUTOSAR_SWS_CANInterface.pdf", "page": 67, "section": "Configuration"}
        ),
        Document(
            page_content="""MESSAGE BUFFERING IN CanIf
            
            Buffer Types:
            1. Reception Buffers: Store received CAN messages before processing
            2. Transmission Buffers: Store messages awaiting transmission
            3. HTH (Hardware Transmit Handle) Buffers: Hardware-specific transmit buffers
            
            Buffer Configuration:
            - Size configurable per message channel
            - Overflow handling strategies defined
            - Priority-based buffer management
            
            Flow Control: Implemented to prevent buffer overflow conditions.""",
            metadata={"source": "AUTOSAR_SWS_CANInterface.pdf", "page": 89, "section": "Buffering"}
        )
    ]
    print(f"   Created {len(documents)} sample documents")

# Text splitting - optimized for technical documents
print("\n2. Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n• ", "\n- ", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)
split_documents = splitter.split_documents(documents)
print(f"   Created {len(split_documents)} chunks")

# Create embeddings and vector store
print("\n3. Creating vector database...")
try:
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Using smaller model for efficiency
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vector_db = FAISS.from_documents(split_documents, embedding)
    print("   Vector database created with OpenAI embeddings")
except Exception as e:
    print(f"   Error with OpenAI embeddings: {e}")
    print("   Using SentenceTransformer embeddings as fallback...")
    try:
        # Install: pip install sentence-transformers
        from langchain.embeddings import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(split_documents, embedding)
        print("   Vector database created with SentenceTransformer embeddings")
    except Exception as e2:
        print(f"   Error with SentenceTransformer: {e2}")
        print("   Using fake embeddings for minimal functionality...")
        from langchain_community.embeddings import FakeEmbeddings
        embedding = FakeEmbeddings(size=384)
        vector_db = FAISS.from_documents(split_documents, embedding)

# Create retriever with MMR for better diversity
retriever = vector_db.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diverse results
    search_kwargs={
        "k": 4,  # Retrieve 4 documents
        "fetch_k": 10,  # Fetch 10 initially for MMR
        "lambda_mult": 0.7  # Balance relevance vs diversity
    }
)

# Create retriever tool with clear instruction
retriever_tool = create_retriever_tool(
    retriever, 
    "autosar_doc_search",
    """Search AUTOSAR CAN Interface specification documents.
    
    USE THIS TOOL FOR:
    - Technical details about software filtering
    - CanIf module architecture and functions
    - Configuration parameters and settings
    - Message handling and buffering
    - AUTOSAR layered architecture details
    
    IMPORTANT: Return verbatim text from documents. Do not interpret or summarize."""
)

tools = [retriever_tool]

# **UPDATED SYSTEM PROMPT - More Directive and Concise**
print("\n4. Creating agent with optimized prompt...")
system_prompt = """You are an AUTOSAR documentation specialist. Your task is to provide ACCURATE, CONCISE answers using ONLY information from the AUTOSAR documents.

**CRITICAL RULES:**
1. **ALWAYS use autosar_doc_search tool FIRST** for every question
2. **Base answers ONLY on retrieved document text** - no external knowledge
3. **Be concise** - answer directly without unnecessary explanations
4. **Cite sources** - mention page/section when available
5. **If information not found**, say "Information not found in AUTOSAR documents"

**ANSWER FORMAT:**
[Direct answer based on documents]
[Optional: Key details from documents]
[Source reference if available]

**EXAMPLE:**
Q: What is software filtering?
A: Software filtering in CanIf allows selective message acceptance based on configurable filter masks (AUTOSAR_SWS_CANInterface.pdf, page 45).

**DOCUMENT FOCUS:** AUTOSAR CAN Interface specifications, configurations, and architecture."""

# Create agent using create_agent
try:
    agent = create_agent(
        model=llm_model,
        tools=tools,
        system_prompt=system_prompt
    )
    print("   Agent created successfully")
except Exception as e:
    print(f"   Error creating agent: {e}")
    print("   Using alternative approach...")
    from langchain.agents import initialize_agent, AgentType
    agent = initialize_agent(
        tools=tools,
        llm=llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=3,
        agent_kwargs={"system_message": system_prompt}
    )

def extract_final_response(result):
    """Extract final AI response from agent result"""
    if isinstance(result, dict):
        if "output" in result:
            return result["output"]
        elif "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'type') and msg.type == "ai" and msg.content:
                    return msg.content
    return str(result)

print("\n5. Testing the agent...")
print("=" * 60)

# Test 1: Software filtering concept
print("\nTest 1: Software filtering concept")
print("-" * 40)

result1 = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is Software filtering in AUTOSAR CAN Interface?"}
    ]
})

response1 = extract_final_response(result1)
print("\nAGENT RESPONSE:")
print(response1)

# Test 2: Specific technical question
print("\n" + "=" * 60)
print("\nTest 2: Filter configuration")
print("-" * 40)

result2 = agent.invoke({
    "messages": [
        {"role": "user", "content": "What are the filter mask configuration parameters in CanIf?"}
    ]
})

response2 = extract_final_response(result2)
print("\nAGENT RESPONSE:")
print(response2)

# Test 3: Architecture question
print("\n" + "=" * 60)
print("\nTest 3: AUTOSAR layers")
print("-" * 40)

result3 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Where does CanIf fit in Classic AUTOSAR architecture?"}
    ]
})

response3 = extract_final_response(result3)
print("\nAGENT RESPONSE:")
print(response3)

# Debug: Show what documents are retrieved
print("\n" + "=" * 60)
print("\nDebug: Document Retrieval Test")
print("-" * 40)

test_queries = [
    "software filtering",
    "filter mask configuration",
    "CanIf architecture layer"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs[:2]):  # Show first 2
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"  Doc {i+1}: {source} (page {page})")
        print(f"  Preview: {doc.page_content[:100]}...")

print("\n" + "=" * 60)
print("AGENT READY FOR USE")
print("=" * 60)

# Interactive mode
print("\nInteractive mode (type 'quit' to exit)")
print("-" * 40)

while True:
    question = input("\nYour question: ").strip()
    if question.lower() in ['quit', 'exit', 'q']:
        break
    
    if not question:
        continue
    
    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })
        
        response = extract_final_response(result)
        print(f"\nAnswer: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to direct retrieval
        docs = retriever.invoke(question)
        if docs:
            print("\nDirect retrieval results:")
            for i, doc in enumerate(docs[:2]):
                print(f"\n[{i+1}] {doc.page_content[:300]}...")
        else:
            print("No information found in documents.")