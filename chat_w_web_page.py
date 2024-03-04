import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter

# load environment variables
load_dotenv()


# check open_ai_api_key in env, if it is not defined as a variable
#it can be added manually in the code below

if os.environ.get("OPENAI_API_KEY") is None or os.environ.get("OPENAI_API_KEY") =="":
    print("open_ai_api_key is not set as environment variable")
else:
    print("Open AI API Key is set")

#get open_ai_api_key
OPEN_AI_API_KEY= os.environ.get("OPEN_AI_API_KEY")


# set tittle for Streamlit UI
st.title("Have a Chat with Web Site")
st.info("Type web site url and ask your questions")


# get web page content
def web_page_context(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Make a research for question and generate a query for creating an information to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


# Create the index using the loaded documents
def web_page_vector(url):
    # get the text from web_page and split then into documents
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store


# Generate a response using OpenAI API
def prompt_template(retriever_chain):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a response for the user's questions based on:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    prompt_retrieval = create_retrieval_chain(retriever_chain, stuff_documents_chain)

    return prompt_retrieval

#get response for the query
def get_response(user_input):
    retriever_chain = web_page_context(st.session_state.vector_store)
    conversation_rag_chain = prompt_template(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

#get web site url from user
website_url = st.text_input("Website URL")

if website_url:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I can help you with your questions related with the web page"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = web_page_vector(website_url)

# user's question
    user_query = st.chat_input("Type your question")
    # Check if query exists in session
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        # Add user's message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Add response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display the existing chat messages from the user and the bot
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
