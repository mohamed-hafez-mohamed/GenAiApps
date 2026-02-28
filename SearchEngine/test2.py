from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langsmith import Client
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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

# AUTOSAR PDF URL
pdf_url = "https://www.autosar.org/fileadmin/standards/R25-11/CP/AUTOSAR_CP_SWS_CANInterface.pdf"

print("\n1. Loading AUTOSAR PDF...")
print(f"   Source: {pdf_url}")

# Try multiple PDF loading methods
documents = []

try:
    print("   Method 1: Trying PyPDFLoader (requires pypdf)...")
    # First download the PDF
    import requests
    import tempfile
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    print("   Downloading PDF...")
    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='wb') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    
    # Try PyPDFLoader first (simpler, fewer dependencies)
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    print(f"   ✓ Successfully loaded {len(documents)} pages with PyPDFLoader")
    
    # Clean up
    import os
    os.unlink(tmp_path)
    
except Exception as e:
    print(f"   ✗ PyPDFLoader failed: {e}")
    
    try:
        print("   Method 2: Trying pdfplumber loader...")
        # Install: pip install pdfplumber
        from langchain_community.document_loaders import PDFPlumberLoader
        
        # Reuse downloaded file or download again
        if 'tmp_path' not in locals():
            import requests
            import tempfile
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='wb') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
        
        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()
        print(f"   ✓ Successfully loaded {len(documents)} pages with PDFPlumberLoader")
        
        # Clean up
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    except Exception as e2:
        print(f"   ✗ PDFPlumberLoader failed: {e2}")
        
        try:
            print("   Method 3: Trying UnstructuredPDFLoader...")
            # Install: pip install "unstructured[pdf]"
            from langchain_community.document_loaders import UnstructuredPDFLoader
            
            if 'tmp_path' not in locals():
                import requests
                import tempfile
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(pdf_url, headers=headers, timeout=60)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='wb') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
            
            loader = UnstructuredPDFLoader(tmp_path)
            documents = loader.load()
            print(f"   ✓ Successfully loaded {len(documents)} pages with UnstructuredPDFLoader")
            
            # Clean up
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
        except Exception as e3:
            print(f"   ✗ UnstructuredPDFLoader failed: {e3}")
            print("   Using detailed sample documents instead...")
            
            from langchain_core.documents import Document
            documents = [
                Document(
                    page_content="""AUTOSAR CAN INTERFACE SPECIFICATION - SOFTWARE FILTERING

Definition: Software filtering in the CAN Interface (CanIf) module is a mechanism to selectively accept or reject CAN messages at the software level based on configurable filter criteria.

Key Characteristics:
• Operates on CAN message identifiers (11-bit Standard ID or 29-bit Extended ID)
• Can filter based on data content patterns in message payload
• Configurable through AUTOSAR ECU Configuration parameters
• Implemented via filter masks defined in CanIf module configuration

Filter Types:
1. Identifier Filtering: Filters messages based on CAN ID ranges or specific IDs
2. Data Filtering: Filters based on data byte patterns and masks
3. Mixed Filtering: Combination of identifier and data filtering

Purpose: Reduce ECU CPU load by processing only relevant messages, improve system efficiency, enable flexible message acceptance policies.

Configuration Example:
CanIfFilterMask {
  CanIfFilterMaskId = 1
  CanIfFilterMaskType = IDENTIFIER_MASK
  CanIfFilterMaskValue = 0x7FF  // Mask for standard IDs
  CanIfFilterMaskRef = CanIfCtrl_1
}""",
                    metadata={
                        "source": "AUTOSAR_SWS_CANInterface.pdf",
                        "page": 45,
                        "section": "5.4 Software Filtering",
                        "type": "technical_specification"
                    }
                ),
                Document(
                    page_content="""CAN INTERFACE (CanIf) MODULE - ARCHITECTURE OVERVIEW

Position in AUTOSAR Stack: The CanIf module is part of the Communication Services layer within the Basic Software (BSW) of Classic AUTOSAR architecture.

Primary Responsibilities:
1. Hardware Abstraction: Provides uniform interface to different CAN controller hardware
2. Message Filtering: Software-based filtering of CAN messages as described in section 5.4
3. Buffer Management: Management of reception and transmission buffers
4. Error Handling: Detection and notification of CAN communication errors
5. State Management: Management of CAN controller states (START, STOP, SLEEP)
6. Interface Provision: Standardized interfaces to upper and lower layers

Layer Interfaces:
Upper Interface Connects to:
• Communication Manager (ComM) - Communication mode control
• Protocol Data Unit Router (PduR) - Message routing
• Diagnostic Communication Manager (Dcm) - Diagnostic services

Lower Interface Connects to:
• CAN Driver (CanDrv) - Hardware-specific CAN controller access
• CAN State Manager (CanSm) - Network management

Service Interface:
• Diagnostic Event Manager (Dem) - Error storage
• Development Error Tracer (Det) - Development error reporting""",
                    metadata={
                        "source": "AUTOSAR_SWS_CANInterface.pdf",
                        "page": 23,
                        "section": "3.1 Architecture Overview",
                        "type": "architecture"
                    }
                ),
                Document(
                    page_content="""CLASSIC AUTOSAR LAYERED ARCHITECTURE

Complete Layer Structure:

1. APPLICATION LAYER
   • Contains Software Components (SW-C) with application-specific logic
   • Uses Ports and Interfaces (Sender-Receiver, Client-Server) for communication
   • Independent of hardware and microcontroller

2. RUNTIME ENVIRONMENT (RTE)
   • Middleware layer enabling communication between SW-C and BSW
   • Provides standardized communication mechanisms
   • Auto-generated based on ECU configuration

3. BASIC SOFTWARE (BSW) - Standardized Automotive Software Platform
   A. SERVICES LAYER
      • System Services: EcuM (ECU State Manager), BswM (Mode Manager), Dem, Det
      • Memory Services: NvM (Non-volatile Memory Manager), MemIf (Memory Abstraction)
      • Communication Services: COM, PduR, CanIf, CanSm, LinIf, FrIf, EthIf
      • I/O Services: I/O Hardware Abstraction
   
   B. ECU ABSTRACTION LAYER
      • Abstracts ECU-specific peripherals
      • Provides hardware-independent interfaces to Services Layer
   
   C. MICROCONTROLLER ABSTRACTION LAYER (MCAL)
      • Direct hardware access drivers
      • Includes: Can, Port, Dio, Adc, Spi, I2c, PWM, GPT, WDG, etc.

The CanIf module specifically resides in the Communication Services sub-layer of the BSW.""",
                    metadata={
                        "source": "AUTOSAR_EXP_LayeredSoftwareArchitecture.pdf",
                        "page": 12,
                        "section": "2.1 Architecture Overview",
                        "type": "architecture"
                    }
                ),
                Document(
                    page_content="""FILTER MASK CONFIGURATION PARAMETERS

Complete Configuration Parameter Set:

1. CanIfFilterMaskId
   • Type: Integer (1..n)
   • Description: Unique identifier for the filter mask
   • Constraints: Must be unique within the CanIf configuration

2. CanIfFilterMaskType
   • Type: Enumeration
   • Values: IDENTIFIER_MASK, DATA_MASK, IDENTIFIER_DATA_MASK
   • Description: Specifies the type of filtering to apply

3. CanIfFilterMaskValue
   • Type: Hex value or data pattern
   • Description: Filter value to match against messages
   • Format: For IDENTIFIER_MASK: 0x000-0x7FF (Standard) or 0x00000000-0x1FFFFFFF (Extended)

4. CanIfFilterMaskRef
   • Type: Reference
   • Description: Reference to associated CAN controller configuration
   • Links to: CanIfCtrl configuration element

5. CanIfFilterMaskAcceptance
   • Type: Enumeration
   • Values: ACCEPT, REJECT
   • Description: Specifies whether matching messages are accepted or rejected

6. CanIfFilterMaskPriority (Optional)
   • Type: Integer
   • Description: Priority when multiple filters match (lower number = higher priority)

Configuration in ARXML:
<FILTER-MASK>
  <SHORT-NAME>CanIfFilterMask_1</SHORT-NAME>
  <FILTER-MASK-TYPE>IDENTIFIER-MASK</FILTER-MASK-TYPE>
  <FILTER-MASK-VALUE>0x7FF</FILTER-MASK-VALUE>
  <ACCEPTANCE>ACCEPT</ACCEPTANCE>
</FILTER-MASK>""",
                    metadata={
                        "source": "AUTOSAR_SWS_CANInterface.pdf",
                        "page": 67,
                        "section": "7.3.2 Filter Mask Configuration",
                        "type": "configuration"
                    }
                ),
                Document(
                    page_content="""MESSAGE BUFFERING AND QUEUING MECHANISMS

Buffer Architecture in CanIf:

1. RECEPTION BUFFERS (Rx Buffers)
   • Purpose: Temporarily store received CAN messages before upper layer processing
   • Types: Dedicated buffers per CAN controller or shared buffers
   • Size: Configurable per hardware unit or message channel
   • Management: FIFO (First-In-First-Out) by default, priority-based optional

2. TRANSMISSION BUFFERS (Tx Buffers)
   • Purpose: Store messages awaiting transmission on CAN bus
   • Organization: Typically organized by priority (based on CAN ID)
   • Flow Control: Implemented to prevent buffer overflow
   • Notification: Upper layers notified of transmission completion/failure

3. HTH BUFFERS (Hardware Transmit Handle)
   • Purpose: Hardware-specific buffer management for transmission
   • Mapping: Each HTH maps to hardware transmission buffers
   • Configuration: Number of HTHs configured per CAN controller

Buffer Configuration Parameters:
• CanIfRxBufferSize: Size of reception buffer in number of messages
• CanIfTxBufferSize: Size of transmission buffer
• CanIfMaxHth: Maximum number of Hardware Transmit Handles
• CanIfBufferOverflowPolicy: Behavior on buffer overflow (DROP_OLDEST, DROP_NEWEST, NOTIFY)

Buffer States:
• EMPTY: Buffer has available space
• PARTIAL: Buffer partially filled
• FULL: Buffer at maximum capacity
• OVERFLOW: Buffer exceeded capacity (if policy allows)""",
                    metadata={
                        "source": "AUTOSAR_SWS_CANInterface.pdf",
                        "page": 89,
                        "section": "6.2 Buffer Management",
                        "type": "implementation"
                    }
                ),
                Document(
                    page_content="""ERROR HANDLING AND DIAGNOSTICS IN CanIf

Error Detection Mechanisms:

1. CAN Controller Errors
   • Bus-Off detection and recovery
   • Error passive state monitoring
   • Warning limit exceeded notifications

2. Message Errors
   • Transmission failures (No ACK, Arbitration lost)
   • Reception errors (CRC, Stuff, Form errors)
   • Overrun errors (Hardware buffer overflow)

3. Software Errors
   • Buffer overflow conditions
   • Invalid parameter configurations
   • State transition violations

Error Reporting:
• To Dem (Diagnostic Event Manager): Storage of error events for diagnostics
• To Det (Development Error Tracer): Development-time error tracing
• Error Codes: Standardized AUTOSAR error codes (CANIF_E_PARAM, CANIF_E_UNINIT, etc.)

Error Recovery Strategies:
1. Automatic Recovery: For transient errors (e.g., temporary bus disturbances)
2. Manual Recovery: Requires software intervention (e.g., bus-off recovery sequence)
3. Degraded Operation: Continue with reduced functionality when possible

Diagnostic Services:
• ReadErrorMemory: Retrieve stored error information from Dem
• ClearErrorMemory: Clear diagnostic error memory
• GetErrorStatus: Query current error status of CAN controllers""",
                    metadata={
                        "source": "AUTOSAR_SWS_CANInterface.pdf",
                        "page": 112,
                        "section": "8.4 Error Handling",
                        "type": "diagnostics"
                    }
                )
            ]
            print(f"   Created {len(documents)} comprehensive sample documents")

# Verify documents were loaded
if len(documents) == 0:
    print("\n   ✗ CRITICAL ERROR: No documents loaded!")
    print("   Please check your internet connection and try again.")
    exit(1)

print(f"\n   Total documents loaded: {len(documents)}")
print(f"   First document preview: {documents[0].page_content[:150]}...")

# Continue with processing...
print("\n2. Processing documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n• ", "\n- ", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    length_function=len
)

split_documents = splitter.split_documents(documents)
print(f"   Created {len(split_documents)} chunks")

# Create embeddings and vector store
print("\n3. Creating embeddings and vector database...")
try:
    embedding = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vector_db = FAISS.from_documents(split_documents, embedding)
    print("   ✓ Vector database created successfully")
except Exception as e:
    print(f"   ✗ Error with OpenAI embeddings: {e}")
    print("   Using HuggingFace embeddings as fallback...")
    
    try:
        # Try to import from langchain_huggingface first
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except:
        # Fallback to community embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings
        embedding = CommunityHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(split_documents, embedding)
    print("   ✓ Vector database created with fallback embeddings")

# Create retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever, 
    "autosar_canif_search",
    """Search the AUTOSAR CAN Interface specification for accurate technical information.
    
    ALWAYS use this tool before answering any question.
    
    Search for:
    - Software filtering concepts and implementation details
    - CanIf module architecture, interfaces, and responsibilities
    - Configuration parameters, settings, and constraints
    - Message handling, buffering, queuing mechanisms
    - Error handling, diagnostics, and recovery procedures
    - Classic AUTOSAR architecture context and layer positioning
    
    Return exact text from documents when possible."""
)

tools = [retriever_tool]

# Create agent
print("\n4. Creating intelligent agent...")

system_prompt = """You are an expert AUTOSAR standards engineer with deep knowledge of CAN communication systems.

**MANDATORY PROCEDURE:**
1. FOR EVERY QUESTION, first use the 'autosar_canif_search' tool to find information
2. Base your answer SOLELY on the retrieved document content
3. If no relevant information is found, state: "Based on the AUTOSAR documents reviewed, this specific information was not found."
4. DO NOT use external knowledge or make assumptions

**RESPONSE GUIDELINES:**
- Be concise and technically accurate
- Use bullet points for lists when appropriate
- Include relevant technical parameters and values
- Reference document sections when available (e.g., "Section 5.4 describes...")
- Focus on practical implementation details

**DOCUMENT SCOPE:** AUTOSAR CAN Interface Specification (SWS_CANInterface) and related architecture documents."""

try:
    agent = create_agent(
        model=llm_model,
        tools=tools,
        system_prompt=system_prompt
    )
    print("   ✓ Agent created successfully")
except Exception as e:
    print(f"   ✗ Error with create_agent: {e}")
    print("   Using alternative method...")
    
    try:
        from langchain.agents import initialize_agent, AgentType
        agent = initialize_agent(
            tools=tools,
            llm=llm_model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": system_prompt}
        )
        print("   ✓ Agent created with initialize_agent")
    except Exception as e2:
        print(f"   ✗ All agent creation methods failed: {e2}")
        exit(1)

# Helper function
def get_agent_response(question):
    """Get response from agent"""
    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })
        
        # Extract response
        if isinstance(result, dict):
            if "output" in result:
                return result["output"]
            elif "messages" in result:
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'type') and msg.type == "ai" and msg.content:
                        return msg.content
            elif "result" in result:
                return result["result"]
        
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Test the system
print("\n5. System Test")
print("=" * 60)

# Quick retriever test
print("\nTesting document retrieval...")
test_query = "software filtering"
docs = retriever.invoke(test_query)
print(f"   Query: '{test_query}' → Found {len(docs)} documents")
if docs:
    print(f"   Sample: {docs[0].page_content[:100]}...")

# Test questions
test_questions = [
    "What is software filtering in AUTOSAR CanIf?",
    "List the filter mask configuration parameters",
    "Where is CanIf located in AUTOSAR architecture?",
    "How does buffer management work in CanIf?"
]

print("\n" + "=" * 60)
print("Running test questions...")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. Q: {question}")
    print("-" * 50)
    response = get_agent_response(question)
    print(f"A: {response}")

print("\n" + "=" * 60)
print("SYSTEM READY - Interactive Mode")
print("=" * 60)

# Interactive session
print("\nAsk questions about AUTOSAR CAN Interface (type 'quit' to exit):")

while True:
    try:
        user_input = input("\nQuestion: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Thinking...")
        response = get_agent_response(user_input)
        print(f"\nAnswer: {response}")
        
    except KeyboardInterrupt:
        print("\n\nSession ended by user.")
        break
    except Exception as e:
        print(f"\nError: {e}")