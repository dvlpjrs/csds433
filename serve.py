import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="HPC Docs Chatbot", layout="wide")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Cache vector store loading for performance
@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


vector_store = load_vector_store()

# Define the custom prompt template
custom_prompt_template = """
Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# Initialize the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4o"),
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={
        "prompt": prompt,  # Pass the properly defined prompt
        "document_variable_name": "context",  # Ensure the variable name matches
    },
)

# Streamlit app layout
st.title("HPC Docs Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = (
        []
    )  # List of tuples (user_message, assistant_message)

# Display existing chat history
for user_msg, assistant_msg in st.session_state["messages"]:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)

# User input
if user_query := st.chat_input("Type your question here..."):
    # Add user query to chat history
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = [
                (user_msg, assistant_msg)
                for user_msg, assistant_msg in st.session_state["messages"]
            ]
            result = qa_chain({"question": user_query, "chat_history": chat_history})
            response = result["answer"]

        st.markdown(response)

    # Add assistant response to chat history
    st.session_state["messages"].append((user_query, response))
