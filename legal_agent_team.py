import streamlit as st
from agno.agent import Agent
# Removed the old PDFKnowledgeBase/PDFReader imports:
# from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat

# Use LangChain's PyPDFLoader for custom PDF ingestion
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import tempfile
import os
from dotenv import load_dotenv
import traceback  # For printing the full traceback

# For Word doc ingestion
import docx2txt

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
QDRANT_URL = os.environ.get('QDRANT_URL')

# ------------------------------------------------------------------------------
# QdrantDoc Class with all required attributes, including 'usage'
# ------------------------------------------------------------------------------
class QdrantDoc:
    """
    Qdrant expects each document to have:
      - document.name       (string)
      - document.content    (string)
      - document.embedding  (list[float])
      - document.meta_data  (dict)
      - document.usage      (dict)
      - document.embed()    (method)
    """
    def __init__(self, doc_id, embedding, payload, content: str):
        self.id = doc_id
        self.name = doc_id
        self.embedding = embedding
        self.meta_data = payload
        self.content = content
        self.usage = {}

    def embed(self, embedder=None):
        # We already have self.embedding, so no-op
        pass

# ------------------------------------------------------------------------------
# Universal Embedder for PDF (string pages) & Word (string)
# ------------------------------------------------------------------------------
class UniversalEmbedder:
    """
    A universal wrapper around langchain.embeddings.OpenAIEmbeddings
    that can embed strings from PDFs or DOCX.
    """
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    def embed(self, text_data):
        text = str(text_data)
        return self.embeddings.embed_documents([text])[0]

# ------------------------------------------------------------------------------
# KnowledgeBase classes for PDF & DOCX
# ------------------------------------------------------------------------------
class CustomPDFKnowledgeBase:
    """
    Stores chunked PDF text in Qdrant, then can do .search(query).
    """
    def __init__(self, vector_db, documents):
        self.vector_db = vector_db
        self.documents = documents

    def search(self, query: str, num_documents: int = 3, **kwargs):
        # Force num_documents to be an int
        if not isinstance(num_documents, int):
            num_documents = 3
        return self.vector_db.search(query=query, limit=num_documents)

class DocxKnowledgeBase:
    """
    Minimal knowledge base for DOC/DOCX, storing doc chunks in Qdrant.
    """
    def __init__(self, vector_db, documents):
        print("Qdrant object methods/attributes (DocxKnowledgeBase init):", dir(vector_db))
        self.vector_db = vector_db
        self.documents = documents

    def search(self, query: str, num_documents: int = 3, **kwargs):
        # Force num_documents to be an int
        if not isinstance(num_documents, int):
            num_documents = 3
        return self.vector_db.search(query=query, limit=num_documents)

# ------------------------------------------------------------------------------
# PDF ingestion function
# ------------------------------------------------------------------------------
def ingest_pdf_document(pdf_path: str, vector_db: Qdrant, embedder: UniversalEmbedder):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # list of Documents with .page_content as string

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separator="\n"
    )

    documents = []
    chunk_id = 0

    for page in pages:
        raw_text = page.page_content
        chunks = text_splitter.split_text(raw_text)

        for chunk_text in chunks:
            doc_id = f"pdf_chunk_{chunk_id}"
            embedding = embedder.embed(chunk_text)

            doc_obj = QdrantDoc(
                doc_id=doc_id,
                embedding=embedding,
                payload={
                    "source": pdf_path,
                    "chunk_index": chunk_id
                },
                content=chunk_text
            )

            vector_db.upsert([doc_obj])
            documents.append({"id": doc_id, "text": chunk_text})
            chunk_id += 1

    return CustomPDFKnowledgeBase(vector_db, documents)

# ------------------------------------------------------------------------------
# Session & Qdrant initialization
# ------------------------------------------------------------------------------
def init_session_state():
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = OPENAI_API_KEY
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = QDRANT_API_KEY
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = QDRANT_URL
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None

def init_qdrant():
    if not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant API key not provided")
    if not st.session_state.qdrant_url:
        raise ValueError("Qdrant URL not provided")
        
    return Qdrant(
        collection="legal_knowledge",
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key,
        https=True,
        timeout=None,
        distance="cosine"
    )

# ------------------------------------------------------------------------------
# Document processing (PDF or DOCX)
# ------------------------------------------------------------------------------
def process_document(uploaded_file, vector_db: Qdrant):
    if not st.session_state.openai_api_key:
        raise ValueError("OpenAI API key not provided")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            embedder = UniversalEmbedder(openai_api_key=st.session_state.openai_api_key)

            if file_ext == ".pdf":
                knowledge_base = ingest_pdf_document(
                    pdf_path=temp_file_path,
                    vector_db=vector_db,
                    embedder=embedder
                )
            elif file_ext in [".doc", ".docx"]:
                raw_text = docx2txt.process(temp_file_path)
                if not raw_text.strip():
                    raise ValueError("No readable text found in the Word document.")

                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                    separator="\n"
                )
                chunks = text_splitter.split_text(raw_text)

                documents = []
                for i, chunk_text in enumerate(chunks):
                    doc_id = f"docx_chunk_{i}"
                    embedding = embedder.embed(chunk_text)
                    
                    doc_obj = QdrantDoc(
                        doc_id=doc_id,
                        embedding=embedding,
                        payload={
                            "source": uploaded_file.name,
                            "chunk_index": i
                        },
                        content=chunk_text
                    )
                    vector_db.upsert([doc_obj])
                    documents.append({"id": doc_id, "text": chunk_text})

                knowledge_base = DocxKnowledgeBase(vector_db, documents)
            else:
                raise ValueError("Unsupported file type. Please upload PDF, DOC, or DOCX.")

            # Quick test
            test_results = knowledge_base.search("test")
            if not test_results:
                raise Exception("Knowledge base verification failed (no search results).")

            return knowledge_base

        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Error processing document: {str(e)}")

# ------------------------------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    try:
        if not st.session_state.vector_db:
            st.session_state.vector_db = init_qdrant()
            st.success("Successfully connected to Qdrant!")
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        return

    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf','doc','docx'])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                st.session_state.knowledge_base = knowledge_base

                # Agents initialization ...
                legal_researcher = Agent(
                    name="Legal Researcher",
                    role="Legal research specialist",
                    model=OpenAIChat(),
                    tools=[DuckDuckGoTools()],
                    knowledge=st.session_state.knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Find and cite relevant legal cases and precedents",
                        "Provide detailed research summaries with sources",
                        "Reference specific sections from the uploaded document",
                        "Always search the knowledge base for relevant information"
                    ],
                    show_tool_calls=True,
                    markdown=True
                )

                contract_analyst = Agent(
                    name="Contract Analyst",
                    role="Contract analysis specialist",
                    model=OpenAIChat(),
                    knowledge=knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Review contracts thoroughly",
                        "Identify key terms and potential issues",
                        "Reference specific clauses from the document"
                    ],
                    markdown=True
                )

                legal_strategist = Agent(
                    name="Legal Strategist", 
                    role="Legal strategy specialist",
                    model=OpenAIChat(),
                    knowledge=knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Develop comprehensive legal strategies",
                        "Provide actionable recommendations",
                        "Consider both risks and opportunities"
                    ],
                    markdown=True
                )

                st.session_state.legal_team = Agent(
                    name="Legal Team Lead",
                    role="Legal team coordinator",
                    model=OpenAIChat(),
                    team=[legal_researcher, contract_analyst, legal_strategist],
                    knowledge=st.session_state.knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Coordinate analysis between team members",
                        "Provide comprehensive responses",
                        "Ensure all recommendations are properly sourced",
                        "Reference specific parts of the uploaded document",
                        "Always search the knowledge base before delegating tasks"
                    ],
                    show_tool_calls=True,
                    markdown=True
                )
                
                st.success("✅ Document processed and team initialized!")
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

        st.divider()
        st.header("🔍 Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Contract Review",
                "Legal Research",
                "Risk Assessment",
                "Compliance Check",
                "Custom Query"
            ]
        )
    else:
        st.warning("Please upload a legal document (PDF, DOC, or DOCX) to begin analysis")

    if not st.session_state.vector_db:
        st.info("👈 Waiting for Qdrant connection...")
    elif not uploaded_file:
        st.info("👈 Please upload a PDF/DOC/DOCX to begin analysis")
    elif st.session_state.legal_team:
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None

        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Please upload a PDF/DOC/DOCX to begin analysis")

if __name__ == "__main__":
    main()