import streamlit as st
from PyPDF2 import PdfReader

import pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage


GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = 'langchain'


def create_pinecone_index():
    pc = pinecone.Pinecone()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # the Google embedding-004 dimensions
            metric="cosine",  # model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        st.info(f"Pinecone index '{INDEX_NAME}' created successfully.")
    else:
        st.warning(f"Pinecone index '{INDEX_NAME}' already exists!")


def delete_pinecone_index():
    pc = pinecone.Pinecone()
    if INDEX_NAME not in pc.list_indexes().names():
        st.warning(f"Not found Pinecone index '{INDEX_NAME}'")
    else:
        pc.delete_index(INDEX_NAME)
        st.info(f"Pinecone index '{INDEX_NAME}' deleted successfully.")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.create_documents([text])
    return chunks


def Pinecone_store_vector_database(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    Pinecone.from_documents(
        text_chunks, embedding=embeddings, index_name=INDEX_NAME)


def get_conversational_chain():
    prompt_template = """
    Please answer the question as detailed and comprehensively as possible based on the given context. Ensure to:
    1. Provide all relevant information from the context
    2. Answer in the same language used in the question
    3. Explain clearly and in a structured manner
    4. Do not omit any important details

    Context:\n{context}\n

    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)
    return response["output_text"]


def main():
    # app config
    st.set_page_config(page_title="Chat PDF", layout='wide', page_icon='🔥')
    # chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello I'm an AI-powered Chatbot. How can I help you?")
        ]
    # user input
    user_question = st.chat_input("Type question about the PDF file here...")
    if user_question is not None and user_input != "":
        response = user_input(user_question)
        st.session_state.chat_history.append(
            HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    # sidebar
    with st.sidebar:
        st.title('Chat with PDF using Gemini🤖')
        st.markdown('''
            A LLM powered ChatBot built using:
            - [Streamlit](https://streamlit.io)
            - [Pinecone](https://pinecone.io)
            - [LangChain](https://python.langchain.com)
            - [GoogleAI](https://ai.google.dev/gemini-api/docs/models/gemini) LLM model
                    ''')
        st.write(
            'The project for presentation for NEWTECH class in [HCMUTE](https://hcmute.edu.vn) 2024-2025')
        st.subheader('©️ Nguyen Duc Huy - 20145449')
        st.subheader('©️ Nguyen The Hao - 21110')

        if st.button("1 - Delete exist Pinecone index"):
            with st.spinner("Processing..."):
                delete_pinecone_index()
                st.balloons()

        if st.button("2 - Create new Pinecone index"):
            with st.spinner("Processing..."):
                create_pinecone_index()
                st.balloons()

        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click the Submit & Process", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                Pinecone_store_vector_database(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()