import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain 


# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set it in the .env file.")

# Set API key for Google Generative AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Language Model and Embeddings
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Function to process and load documents into a vector store
def create_vector_store():
    try:
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        #     temp_file.write(pdf_file.read())
        #     temp_file_path = temp_file.name

        loader = PyPDFLoader("./Updated_structred_new.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./Gemini_data"
        )
        logging.info(f"Vector store created with {len(document_chunks)} chunks.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        logging.error(f"Error in create_vector_store: {e}")
        return None



def create_contextual_chain(vector_store):
    """Creates a contextual retriever chain."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})


        # Define the prompt for contextualizing the question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = llm_gemini | StrOutputParser()

    # Create a history-aware retriever
    retriever_chain = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return retriever_chain

def get_conversational_chain(retriever_chain):
    # Define the prompt for the question-answering chain
    rag_prompt = (
       """ Hello! I'm your PreCollege AI assistant. I'll guide you through your JEE Mains journey, providing personalized advice and support.
        To get started, please share your JEE Mains rank and preferred engineering branches or colleges. 
        I'll provide information and suggestions based on our database.Please note that I'll only provide information available within our database, ensuring accuracy and relevance. Let's get started!"""
        "\n\n"
        "{context}")

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the question-answering chain
    document_chain = create_stuff_documents_chain(llm_gemini, qa_prompt)
    # Create the retrieval-augmented generation (RAG) chain
    return create_retrieval_chain(retriever_chain, document_chain)


def get_response(user_query):
    # Define a set of common greetings
    greetings = ["hi", "hello", "hey", "greetings", "hi there"]

    # Check if the user query is a greeting
    if user_query.lower().strip() in greetings:
        return "Hello! How can I assist you with your college search today?"

    # Ensure the vector store is available in the session state
    if "vector_store" not in st.session_state:
        return "Vector store is not initialized. Please upload and preprocess a document."

    # Create the history-aware retriever chain
    retriever_chain = create_contextual_chain(st.session_state.vector_store)
    rag_chain = get_conversational_chain(retriever_chain)


    # Format the chat history
    formatted_chat_history = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            formatted_chat_history.append({"author": "user", "content": message.content})
        elif isinstance(message, SystemMessage):
            formatted_chat_history.append({"author": "assistant", "content": message.content})

    # Invoke the RAG chain with the user query and chat history
    response = rag_chain.invoke({
        "chat_history": formatted_chat_history,
        "input": user_query
    })

    # Update the chat history in the session state
    st.session_state.chat_history.append({"author": "user", "content": user_query})
    st.session_state.chat_history.append({"author": "assistant", "content": response['answer']})

    return response['answer']

# st.session_state.vector_store = create_vector_store()

import streamlit as st

# Initialize the vector store and store it in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_vector_store()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI setup
st.set_page_config(page_title="College Data Chatbot")
st.title("Gemini Chatbot")

# Main app logic
if st.session_state.vector_store is None:
    st.error("Failed to load preprocessed data. Please ensure the data exists in the specified directory.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["author"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Append user message to chat history
        # st.session_state.chat_history.append({"author": "user", "content": prompt})
        # Generate assistant response
        response = get_response(prompt)
        # Append assistant response to chat history
        # st.session_state.chat_history.append({"author": "assistant", "content": response})
        # Rerun the app to display the updated chat history
        st.rerun()