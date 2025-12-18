import streamlit as st
import requests
import json
from PyPDF2 import PdfReader
import tempfile
import os
from typing import Optional

class OllamaPDFSummarizer:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """Extract text content from PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def summarize_text(self, text: str, model: str = "llama3.2:latest") -> Optional[str]:
        """Summarize text using Ollama"""
        try:
            # Limit text length to avoid context window issues
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                st.warning(f"Text truncated to {max_chars} characters for processing.")
            
            prompt = f"""Please provide a comprehensive summary of the following text. 
            Focus on the main points, key arguments, and important conclusions:

            {text}

            Summary:"""
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            with st.spinner("Generating summary..."):
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No summary generated.')
            else:
                st.error(f"Error from Ollama: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again with a shorter document.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="PDF Summarizer with Ollama",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Summarizer with Ollama")
    st.markdown("Upload a PDF document and get an AI-powered summary using Ollama!")
    
    # Initialize summarizer
    summarizer = OllamaPDFSummarizer()
    
    # Check Ollama connection
    with st.sidebar:
        st.header("🔧 Connection Status")
        
        # Allow custom Ollama host configuration
        ollama_host = st.text_input("Ollama Host:", value="http://localhost:11434", help="Change to http://77.37.45.138:11434 for external access")
        if ollama_host != summarizer.base_url:
            summarizer = OllamaPDFSummarizer(ollama_host)
        
        if summarizer.check_ollama_connection():
            st.success("✅ Ollama is connected!")
            
            # Get available models
            models = summarizer.get_available_models()
            if models:
                selected_model = st.selectbox("Select Model:", models)
                st.info(f"Using model: {selected_model}")
            else:
                st.error("No models available!")
                return
        else:
            st.error("❌ Cannot connect to Ollama!")
            st.markdown("""
            *Troubleshooting:*
            1. Make sure Ollama is running: ollama serve
            2. Check if accessible: curl http://localhost:11434/api/tags
            3. For external access, set: OLLAMA_HOST=0.0.0.0
            """)
            return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF document to summarize"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Extract text
            with st.spinner("Extracting text from PDF..."):
                text = summarizer.extract_text_from_pdf(uploaded_file)
            
            if text:
                st.info(f"Extracted {len(text)} characters from PDF")
                
                # Show text preview
                with st.expander("📖 Preview extracted text"):
                    st.text_area("Text content:", text[:1000] + "..." if len(text) > 1000 else text, height=200)
                
                # Summarize button
                if st.button("🚀 Generate Summary", type="primary"):
                    summary = summarizer.summarize_text(text, selected_model)
                    
                    if summary:
                        st.session_state['summary'] = summary
                        st.session_state['original_text'] = text
    
    with col2:
        st.header("📝 Summary")
        
        if 'summary' in st.session_state:
            st.success("Summary generated successfully!")
            
            # Display summary
            summary_container = st.container()
            with summary_container:
                st.text_area(
                    "AI Summary:", 
                    st.session_state['summary'], 
                    height=400,
                    help="AI-generated summary of your PDF"
                )
            
            # Summary statistics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                original_words = len(st.session_state['original_text'].split())
                st.metric("Original Words", original_words)
            
            with col2_2:
                summary_words = len(st.session_state['summary'].split())
                st.metric("Summary Words", summary_words)
            
            # Compression ratio
            if original_words > 0:
                compression_ratio = (1 - summary_words / original_words) * 100
                st.metric("Compression", f"{compression_ratio:.1f}%")
            
            # Download summary
            st.download_button(
                label="💾 Download Summary",
                data=st.session_state['summary'],
                file_name="pdf_summary.txt",
                mime="text/plain"
            )
        else:
            st.info("Upload a PDF and click 'Generate Summary' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Ollama* 🚀")

if __name__ == "__main__":
    main()