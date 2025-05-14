import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub






def get_pdf_text(pdf_docs):
    text=" "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings()
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    # Create a vector store from the text chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

 
# Initialize model
def get_conversation_chain(vectorstore):

    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
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
    # Load environment variables from .env file
    load_dotenv()
    # Set the page configuration

    st.set_page_config(page_title="My Streamlit App", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "conversation" not in st.session_state:
        st.session_state.chat_history= None
 
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
    st.markdown(user_template.replace("{{MSG}}","hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","hello human"),unsafe_allow_html=True)



   


    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs=st.file_uploader("Upload a file",accept_multiple_files=True)
        if st.button("Processing"):
            with st.spinner("Processing..."):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get the text chunks
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)

                #create vector store
                
                vectorstore=get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
                
if __name__ == "__main__":
    main()