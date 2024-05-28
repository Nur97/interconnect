import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_python_code_info(py_files):
    code_info = []
    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            code_info.append((file_path, file_content))
            #st.write(code_info)
    return code_info


def get_code_chunks(code_info):
    chunks = []
    for info in code_info:
        chunks.append(info)
        #file_path, file_content = info 
        #chunks.append((file_path, file_content))
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #texts = [chunk[0] + "\n" + chunk[1] for chunk in text_chunks]
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    #return vectorstore
    embeddings = OpenAIEmbeddings()
    texts = []
    for chunk in text_chunks:
        file_path, code_content = chunk
        text = file_path + "\n" + code_content
        texts.append(text)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
  
    # Check for cached data and use it if available
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.subheader("Your PHP Files")
    st.header("Chat with multiple PHP files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
       handle_userinput(user_question)

    with st.sidebar:
      #try:
        path_input = st.text_input("Enter the path to the directory containing Python files:")
        if path_input:
            if os.path.exists(path_input) and os.path.isdir(path_input):
                py_files = []
                for root, dirs, files in os.walk(path_input):
                    for file in files:
                        if file.endswith('.php') or file.endswith('.js') or file.endswith('.css') or file.endswith('.html') or file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            py_files.append(file_path)
                            st.write(f"Added file to py_files: {file_path}")
            else:
                raise FileNotFoundError("Invalid directory path. Please enter a valid path.")
        if st.button("Handel New Files"):
            with st.spinner("Processing PHP code..."):
                raw_code_info = get_python_code_info(py_files)
                code_chunks = get_code_chunks(raw_code_info)
                text_chunks_with_paths = [(chunk[0], chunk[1]) for chunk in code_chunks]
                vectorstore = get_vectorstore(text_chunks_with_paths)
                st.session_state.conversation = get_conversation_chain(vectorstore)
if __name__ == '__main__':
    main()
