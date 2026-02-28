import streamlit as st
import time
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Environment setup ────────────────────────────────────────────────
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROK_API_KEY"]    = os.getenv("GROK_API_KEY")

# LangSmith (optional)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-pdf-app")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY_SIMPLE_RAG_DOCUMENT")

groq_api_key = os.getenv("GROK_API_KEY")
llm_model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# ── Session state initialization ─────────────────────────────────────
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

# ── Helpers ──────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided document only.
    Provide the most accurate answer possible from the given context.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# ── Vector store creation ────────────────────────────────────────────
def create_vector_embeddings(uploaded_file):
    if uploaded_file is None:
        st.warning("No file uploaded.")
        return

    # Skip if already done
    if (st.session_state.processed_file_name == uploaded_file.name and
        st.session_state.vectors is not None):
        st.info("Document already processed.")
        return

    with st.spinner("Processing large PDF (this may take several minutes)..."):
        # ── Temp file save ──
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            st.caption(f"Loaded {len(documents)} pages")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,          # smaller chunks = safer & cheaper
                chunk_overlap=100,       # some overlap helps with context continuity
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            st.caption(f"Created {len(chunks)} chunks")
            embeddings = (OllamaEmbeddings(model = "mxbai-embed-large")) 

            # ── MANUAL BATCH EMBEDDING ── most important change
            batch_size = 8             
            vector_store = None

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                status_text.text(f"Embedding batch {i//batch_size + 1} of {len(chunks)//batch_size + 1} ({len(batch)} chunks)...")

                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)

                # Update progress
                progress = (i + len(batch)) / len(chunks)
                progress_bar.progress(progress)

                time.sleep(0.5)   # tiny delay — helps avoid hammering rate limits

            st.session_state.vectors = vector_store
            st.session_state.processed_file_name = uploaded_file.name

            st.success(f"Vector store created successfully! ({len(chunks)} chunks embedded)")

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            if "Connection" in str(e) or "timeout" in str(e).lower():
                st.warning(
                    "Connection dropped during large embedding. "
                    "Try again, or use mobile hotspot / different network. "
                    "Large documents are sensitive to network stability."
                )
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ── UI ───────────────────────────────────────────────────────────────
st.title("PDF Q&A with RAG")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if st.button("Process Document", disabled=uploaded_file is None, type="primary"):
    create_vector_embeddings(uploaded_file)

# ── Question & Answer area ───────────────────────────────────────────
if st.session_state.vectors is not None:
    st.subheader("Ask a question about the document")

    user_prompt = st.text_input("Your question:", key="question_input")

    if user_prompt:
        with st.spinner("Searching & generating answer..."):
            retriever = st.session_state.vectors.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "input": RunnablePassthrough()
                }
                | prompt
                | llm_model
                | StrOutputParser()
            )

            start_time = time.process_time()
            answer = rag_chain.invoke(user_prompt)
            elapsed = time.process_time() - start_time

            st.markdown("**Answer:**")
            st.write(answer)
            st.caption(f"⏱️ Generated in {elapsed:.2f} seconds")

            # Show retrieved chunks
            with st.expander("Retrieved context chunks"):
                docs = retriever.invoke(user_prompt)
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**  (source: page ~{doc.metadata.get('page', '?')})")
                    st.markdown(doc.page_content)
                    st.markdown("---")

else:
    st.info("Upload a PDF and click **Process Document** to start asking questions.")