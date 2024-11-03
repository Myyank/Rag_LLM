import streamlit as st
import PyPDF2
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the Groq model
API_KEY = "API_KEY"  # Replace with your actual API key
llm = Groq(model="llama-3.1-70b-versatile", api_key=API_KEY)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Streamlit app layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("Welcome to Chat with RAG Assistant")

with st.sidebar:
    st.header("Upload PDF Documents")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Function to convert PDF files to text
def process_pdf_files(files):
    documents = []
    for file in files:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        # Wrap text in a Document object
        document = Document(text=text)
        documents.append(document)
    return documents

# Process documents
if uploaded_files:
    documents = process_pdf_files(uploaded_files)
    st.write("Documents processed and ready for indexing.")

    index = VectorStoreIndex.from_documents(documents)

    # Configure retriever and response synthesizer
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

# Initialize a session state variable to hold responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# User input for the chat
user_input = st.chat_input('Message to Assistant...', key='prompt_input')

# Handle user input and generate response
if user_input:
    with st.spinner("Generating response..."):
        if uploaded_files:
            context = query_engine.query(user_input)
            user_input = f"question: {user_input} context: {context}"

        response = llm.stream_complete(user_input)

        # Display generated response
        response_placeholder = st.empty()  
        streamed_response = ""
        for chunk in response:
            streamed_response += chunk.delta or ""
            response_placeholder.write(streamed_response)

        # Store the generated response
        st.session_state.responses.append((user_input, streamed_response))


st.write("Generated Responses:")
for question, answer in st.session_state.responses:
    st.write(f"**Q:** {question}")
    st.write(f"**A:** {answer}")
