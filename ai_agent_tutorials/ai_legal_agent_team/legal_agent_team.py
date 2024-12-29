# app.py
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from typing import List
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state variables"""
    session_vars = {
        'openai_api_key': None,
        'qdrant_api_key': None,
        'qdrant_url': None,
        'vector_db': None,
        'qa_chain': None,
        'uploaded_files': [],
        'processed_files': set()
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def init_qdrant():
    """Initialize Qdrant vector database"""
    if not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant API key not provided")
    if not st.session_state.qdrant_url:
        raise ValueError("Qdrant URL not provided")
        
    try:
        client = QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key
        )
        
        # Create collection if it doesn't exist
        collection_name = "legal_docs"
        collections = client.get_collections().collections
        if not any(collection.name == collection_name for collection in collections):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
        vector_db = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        logger.info("Successfully connected to Qdrant")
        return vector_db
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")
        raise

def process_documents(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], vector_db: Qdrant):
    """Process documents and add to vector store"""
    if not uploaded_files:
        raise ValueError("No files uploaded")
        
    st.write("Starting document processing...")  # Debug output
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        all_docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                    logger.warning(f"Skipping {uploaded_file.name}: File size exceeds 200MB limit")
                    continue
                
                # Save uploaded file temporarily
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and split the PDF
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                split_docs = text_splitter.split_documents(documents)
                all_docs.extend(split_docs)
                logger.info(f"Processed file: {uploaded_file.name}")
        
                        # Add documents to vector store
        if all_docs:
            st.write(f"Processing {len(all_docs)} document chunks...")  # Debug output
            vector_db.add_documents(all_docs)
            st.write("✅ Documents added to vector store")  # Debug output
            logger.info(f"Added {len(all_docs)} document chunks to vector store")
        else:
            st.warning("No document chunks were created. Please check the PDF content.")
        
        return vector_db
        
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def init_qa_chain(vector_db: Qdrant):
    """Initialize the QA chain with conversation memory"""
    try:
        # Initialize language model
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0,
            api_key=st.session_state.openai_api_key
        )
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Configure retriever with search parameters
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )
        
        # Create retrieval chain with more specific configuration
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,  # Add verbose output for debugging
            chain_type="stuff"  # Use 'stuff' method for document compilation
        )
        
        logger.info("Successfully initialized QA chain")
        return qa_chain
    
    except Exception as e:
        logger.error(f"Error initializing QA chain: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Document Analyzer 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
   
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key from your Qdrant Cloud dashboard"
        )
        # Show help text for Qdrant key
        st.caption("You can find your API key in the Qdrant Cloud dashboard under API keys section")
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "https://69792e3c-ca29-4963-983e-b6d9803d0a76.us-east4-0.gcp.cloud.qdrant.io:6333",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    st.session_state.vector_db = init_qdrant()
                    st.success("Successfully connected to Qdrant!")
            except Exception as e:
                st.error(f"Failed to connect to Qdrant: {str(e)}")

        st.divider()

        # Document upload and processing section
        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            
            uploaded_files = st.file_uploader(
                "Upload Legal Documents", 
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                if new_files:
                    with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                        try:
                            vector_db = process_documents(new_files, st.session_state.vector_db)
                            st.session_state.qa_chain = init_qa_chain(vector_db)
                            st.session_state.processed_files.update(f.name for f in new_files)
                            st.success(f"✅ {len(new_files)} new document(s) processed!")
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
            
            # Knowledge base management section
            st.divider()
            st.header("📚 Document Management")
            
            if st.session_state.processed_files:
                st.write(f"Total Documents: {len(st.session_state.processed_files)}")
                
                for idx, filename in enumerate(sorted(st.session_state.processed_files), 1):
                    with st.expander(f"📄 {filename}"):
                        st.write(f"Document #{idx}")
                        if st.button(f"Remove {filename}", key=f"remove_{filename}"):
                            st.session_state.processed_files.remove(filename)
                            st.experimental_rerun()
                
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    st.session_state.processed_files.clear()
                    st.session_state.qa_chain = None
                    st.session_state.vector_db = init_qdrant()  # Reinitialize empty vector store
                    st.experimental_rerun()
            else:
                st.info("No documents uploaded yet")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not st.session_state.processed_files:
        st.info("👈 Please upload at least one legal document to begin analysis")
    elif st.session_state.qa_chain:
        st.header("🔍 Legal Document Analysis")
        
        # Query templates
        query_templates = {
            "Document Comparison": "Compare and analyze the relationships between all uploaded documents. Identify any conflicts, overlaps, or dependencies.",
            "Contract Review": "Review all contracts and identify key terms, obligations, and potential issues.",
            "Legal Research": "Research and identify relevant cases and precedents related to these documents.",
            "Risk Assessment": "Analyze potential legal risks and liabilities across all documents.",
            "Compliance Check": "Check all documents for regulatory compliance issues.",
            "Custom Query": None
        }
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            list(query_templates.keys())
        )
        
        # Query input
        if analysis_type == "Custom Query":
            query = st.text_area("Enter your custom query:", height=100)
        else:
            query = query_templates[analysis_type]
            st.info(f"Template query: {query}")
            if st.checkbox("Customize this query"):
                query = st.text_area("Edit query:", value=query, height=100)
        
        # Run analysis
        if query and st.button("Run Analysis"):
            with st.spinner("Analyzing documents..."):
                try:
                    # Add analysis context to the query
                    enhanced_query = f"""
                    Using ONLY the content from the provided documents, please analyze the following query:
                    {query}
                    
                    Provide a detailed analysis that includes:
                    1. Key findings and insights with specific references to the documents
                    2. Direct quotes from the documents to support your analysis
                    3. Potential implications or recommendations based on the document content
                    4. Any areas requiring further analysis
                    
                    If you cannot find relevant information in the documents, please specifically indicate what information is missing.
                    Remember to ONLY use information that is explicitly present in the provided documents.
                    """
                    
                    st.write("Searching through documents...")  # Debug output
                    
                    response = st.session_state.qa_chain({"question": enhanced_query})
                    
                    # Display response
                    st.markdown("### Analysis Results")
                    st.markdown(response["answer"])
                    
                    # Display sources if available
                    if response.get("source_documents"):
                        with st.expander("📚 Source References"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(doc.page_content)
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()