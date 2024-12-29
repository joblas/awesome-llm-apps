import streamlit as st
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.qdrant import Qdrant
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
import tempfile
import os

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

def init_qdrant():
    """Initialize Qdrant vector database"""
    if not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant API key not provided")
    if not st.session_state.qdrant_url:
        raise ValueError("Qdrant URL not provided")
        
    try:
        vector_db = Qdrant(          
            collection="legal_knowledge",
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            https=True,
            timeout=60,
            distance="cosine"
        )
        
        vector_db.client.get_collections()
        return vector_db
    except Exception as e:
        raise Exception(f"Failed to initialize Qdrant: {str(e)}")

def process_documents(uploaded_files, vector_db: Qdrant):
    """Process multiple documents, create embeddings and store in Qdrant vector database"""
    if not st.session_state.openai_api_key:
        raise ValueError("OpenAI API key not provided")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            if not uploaded_files:
                raise ValueError("No files uploaded")
            
            # Save all files to temporary directory
            for uploaded_file in uploaded_files:
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                    st.warning(f"Skipping {uploaded_file.name}: File size exceeds 200MB limit")
                    continue
                    
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                api_key=st.session_state.openai_api_key
            )
            
            try:
                knowledge_base = PDFKnowledgeBase(
                    path=temp_dir, 
                    vector_db=vector_db, 
                    reader=PDFReader(chunk=True),
                    embedder=embedder,
                    recreate_vector_db=True  
                )
                knowledge_base.load()
                return knowledge_base
                
            except Exception as kb_error:
                raise Exception(f"Error creating knowledge base: {str(kb_error)}")
                
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}\n"
            if hasattr(e, 'response'):
                error_msg += f"Response status: {e.response.status_code}\n"
                error_msg += f"Response content: {e.response.content}"
            raise Exception(error_msg)

def init_legal_team(knowledge_base):
    """Initialize the legal team with given knowledge base"""
    legal_researcher = Agent(
        name="Legal Researcher",
        role="Legal research specialist",
        model=OpenAIChat(model="gpt-4o"),
        tools=[DuckDuckGo()],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Find and cite relevant legal cases and precedents",
            "Provide detailed research summaries with sources",
            "Reference specific sections from all uploaded documents",
            "Always search the knowledge base for relevant information"
        ],
        show_tool_calls=True,
        markdown=True
    )

    contract_analyst = Agent(
        name="Contract Analyst",
        role="Contract analysis specialist",
        model=OpenAIChat(model="gpt-4o"),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Review all contracts thoroughly",
            "Identify key terms and potential issues",
            "Cross-reference clauses between documents",
            "Reference specific clauses from all documents"
        ],
        markdown=True
    )

    legal_strategist = Agent(
        name="Legal Strategist", 
        role="Legal strategy specialist",
        model=OpenAIChat(model="gpt-4o"),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Develop comprehensive legal strategies",
            "Provide actionable recommendations",
            "Consider both risks and opportunities",
            "Analyze relationships between multiple documents"
        ],
        markdown=True
    )

    return Agent(
        name="Legal Team Lead",
        role="Legal team coordinator",
        model=OpenAIChat(model="gpt-4o"),
        team=[legal_researcher, contract_analyst, legal_strategist],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Coordinate analysis between team members",
            "Provide comprehensive responses",
            "Ensure all recommendations are properly sourced",
            "Reference specific parts of all uploaded documents",
            "Always search the knowledge base before delegating tasks",
            "Consider relationships and dependencies between documents"
        ],
        show_tool_calls=True,
        markdown=True
    )

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

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
            help="Enter your Qdrant API key"
        )
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

        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            
            # Multiple file uploader
            uploaded_files = st.file_uploader(
                "Upload Legal Documents", 
                type=['pdf'],
                accept_multiple_files=True
            )
            
            # Track uploaded files in session state
            if uploaded_files:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                if new_files:
                    with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                        try:
                            knowledge_base = process_documents(new_files, st.session_state.vector_db)
                            st.session_state.knowledge_base = knowledge_base
                            st.session_state.legal_team = init_legal_team(knowledge_base)
                            
                            # Update processed files list
                            st.session_state.processed_files.update(f.name for f in new_files)
                            st.success(f"✅ {len(new_files)} new document(s) processed!")
                            
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")

            st.divider()
            st.header("📚 Knowledge Base")
            
            # Create expandable section for document details
            if st.session_state.processed_files:
                # Display document count
                doc_count = len(st.session_state.processed_files)
                st.write(f"Total Documents: {doc_count}")
                
                # Create expandable section for each document
                for idx, filename in enumerate(sorted(st.session_state.processed_files), 1):
                    with st.expander(f"📄 {filename}"):
                        st.write(f"Document #{idx} in Knowledge Base")
                        if st.button(f"Remove {filename}", key=f"remove_{filename}"):
                            st.session_state.processed_files.remove(filename)
                            st.experimental_rerun()
                
                # Add option to clear all documents
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    st.session_state.processed_files.clear()
                    st.session_state.knowledge_base = None
                    st.session_state.legal_team = None
                    st.experimental_rerun()
            else:
                st.info("No documents in knowledge base yet")
            
            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Document Comparison",
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not st.session_state.processed_files:
        st.info("👈 Please upload at least one legal document to begin analysis")
    elif st.session_state.legal_team:
        # Analysis icons dictionary
        analysis_icons = {
            "Document Comparison": "🔄",
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        st.header(f"{analysis_icons[analysis_type]} {analysis_type}")
  
        analysis_configs = {
            "Document Comparison": {
                "query": "Compare and analyze the relationships between all uploaded documents. Identify any conflicts, overlaps, or dependencies.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Cross-document analysis and comparison"
            },
            "Contract Review": {
                "query": "Review all contracts and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to these documents.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities across all documents.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check all documents for regulatory compliance issues.",
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
        st.write(f"🤖 Active Legal AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

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
                with st.spinner("Analyzing documents..."):
                    try:
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using all uploaded documents as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from all documents.
                            """
                        else:
                            combined_query = f"""
                            Using all uploaded documents as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from all documents.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
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
        st.info("Please upload a legal document to begin analysis")

if __name__ == "__main__":
    main()
