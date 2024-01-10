import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ''
    pdfReader = PdfReader(pdf_docs)
    for page in pdfReader.pages:
        text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """Answer the question as detailes as possible from the provided context, make sure to provide all the details, if the answer is not in
                        provided in the context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
                        Context:\n{context}\n
                        Question: \n{question}\n

                        Answer:  
                        """
    # st.write('get conversational chain function')
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # st.write('get conversational chain function 2')
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    # st.write('get conversational chain function 3')
    chain = load_qa_chain(model, chain_type = 'stuff', prompt=prompt)
    # st.write('get conversational chain function 4')

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings)
    docs = new_db.similarity_search(user_question)  
    # st.write('hari')

    chain = get_conversational_chain()

    response = chain(
                {'input_documents' : docs, 'question':user_question},
                return_only_outputs=True)
    print(response)
    st.write("Reply: ", response['output_text'])

# MAX_FILE_SIZE_MB = 200

def main():
    # pass
    st.sidebar.title("Menu")
    uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", accept_multiple_files=True)
    submit_button = st.sidebar.button("Submit & Process")

    st.title("Multi PDF Chat with Gemini")
    user_question = st.text_input("Ask your question here")

    if user_question:
        # Process user input (you can add your backend logic here)
        if uploaded_files:
            st.write(f"User input: {user_question}")
            user_input(user_question)
            # Add backend processing and display responses here
        else:
            st.write('Please check wether you forgot to Upload pdfs...')


    if submit_button:
        if uploaded_files:
            # Process uploaded PDFs (you can add your backend logic here)
            raw_text = ''
            text_chunks = ''
            for pdf_file in uploaded_files:
                # Check if the file is a PDF
                if pdf_file.type == "application/pdf":
                    raw_text += get_pdf_text(pdf_file)
                    # pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                    # num_pages = pdf_reader.numPages
                    # st.write(f"Uploaded: {pdf_file.name} - Number of Pages: {num_pages}")
                    # Perform operations with the PDF file here (e.g., count pages, extract text, etc.)
            text_chunks= get_text_chunks(raw_text)
            # st.write(text_chunks)
            get_vector_store(text_chunks)
            st.success('Files processed successfully')
            # with st.spinner('Processing...'):
            #     raw_text = get_pdf_text(uploaded_files)
            #     text_chunks = get_text_chunks(raw_text)
            #     get_vector_store(text_chunks)
            #     st.success('Done')
            # for pdf_file in uploaded_files:
            #     # Example: Display the names of the uploaded PDF files
            #     st.sidebar.write(f"Uploaded: {pdf_file.name}")
            #     raw_text = get_pdf_text(pdf_file)
   
if __name__ == "__main__":
    main()

