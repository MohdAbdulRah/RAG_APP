import streamlit as st
from rag import preprocess_urls, generate_answer, initialize_components

# Page configuration
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_urls" not in st.session_state:
    st.session_state.current_urls = []

# Title and description
st.title("🤖 RAG-Based Question Answering System")
st.markdown("Upload URLs and ask questions based on their content using RAG (Retrieval Augmented Generation)")

# Sidebar for URL input
with st.sidebar:
    st.header("📚 Document Sources")
    
    # URL input area
    st.subheader("Enter URLs")
    url_input = st.text_area(
        "Add URLs (one per line)",
        height=200,
        placeholder="https://example.com\nhttps://another-site.com",
        help="Enter one URL per line. These will be processed and indexed."
    )
    
    # Process button
    if st.button("🔄 Process URLs", type="primary", use_container_width=True):
        if url_input.strip():
            # Parse URLs
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            if urls:
                with st.spinner("Processing URLs... This may take a moment."):
                    try:
                        # Initialize and process
                        initialize_components()
                        preprocess_urls(urls)
                        
                        # Update session state
                        st.session_state.urls_processed = True
                        st.session_state.current_urls = urls
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ Successfully processed {len(urls)} URL(s)!")
                    except Exception as e:
                        st.error(f"❌ Error processing URLs: {str(e)}")
            else:
                st.warning("⚠️ Please enter at least one valid URL")
        else:
            st.warning("⚠️ Please enter URLs to process")
    
    # Display current sources
    if st.session_state.urls_processed and st.session_state.current_urls:
        st.divider()
        st.subheader("📄 Indexed Sources")
        for i, url in enumerate(st.session_state.current_urls, 1):
            st.text(f"{i}. {url[:50]}...")
    
    # Clear button
    if st.session_state.urls_processed:
        st.divider()
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.urls_processed = False
            st.session_state.chat_history = []
            st.session_state.current_urls = []
            st.rerun()

# Main content area
if not st.session_state.urls_processed:
    # Welcome screen
    st.info("👈 Please enter URLs in the sidebar and click 'Process URLs' to get started")
    
    st.markdown("### 🚀 How to use:")
    st.markdown("""
    1. **Enter URLs** in the sidebar (one per line)
    2. **Click 'Process URLs'** to index the content
    3. **Ask questions** about the content
    4. **Get answers** with source citations
    """)
    
    st.markdown("### 💡 Example URLs:")
    st.code("""https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Deep_learning""")

else:
    # Question answering interface
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat["sources"]:
                with st.expander("📚 View Sources"):
                    for source in chat["sources"]:
                        st.markdown(f"- {source}")
    
    # Question input
    question = st.chat_input("Ask a question about the documents...")
    
    if question:
        # Display user question
        with st.chat_message("user"):
            st.write(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = generate_answer(question)
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                st.markdown(f"- {source}")
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by LangChain, Groq, and ChromaDB</p>",
    unsafe_allow_html=True
)