import sys
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from sentence_transformers.util import semantic_search
import requests
from pathlib import Path
from time import sleep
import torch
import os
import random
import string
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="USinoIP Website AI Chat Assistant", layout="wide")
st.subheader("Welcome to USinoIP Website AI Chat Assistant.")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')
#HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
#model_id = os.environ.get('model_id')
#hf_token = os.environ.get('hf_token')
#repo_id = os.environ.get('repo_id')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def get_embeddings(input_str_texts):
    response = requests.post(api_url, headers=headers, json={"inputs": input_str_texts, "options":{"wait_for_model":True}})
    return response.json()

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

prompt_template = """
#You are a very helpful AI assistant. Please ONLY use {context} to answer the user's input question. If you don't know the answer, just say that you don't know. DON'T try to make up an answer and do NOT go beyond the given context without the user's explicitly asking you to do so!
You are a very helpful AI assistant. Please response to the user's input question with as many details as possible.
Question: {question}
Helpful AI Repsonse:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))  
    
#url="https://www.usinoip.com"
url="https://www.usinoip.com/UpdatesAbroad/290.html"
texts=""
raw_text=""
user_question = ""
initial_embeddings=""
db_embeddings = ""
i_file_path=""
file_path = ""
random_string=""
wechat_image= "WeChatCode.jpg"

st.sidebar.markdown(
    """
    <style>
    .blue-underline {
        text-decoration: bold;
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }
    </style>
    """, unsafe_allow_html=True
)

user_question = st.text_input("Enter your query here and AI-Chat with your website:")

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

with st.sidebar:
    st.subheader("You are chatting with USinoIP official website.")
    st.write("Note & Disclaimer: This app is provided on open source framework and is for information purpose only. NO guarantee is offered regarding information accuracy. NO liability could be claimed against whoever associated with this app in any manner. User should consult a qualified legal professional for legal advice.")
    st.sidebar.markdown("Contact: [aichat101@foxmail.com](mailto:aichat101@foxmail.com)")
    st.sidebar.markdown('WeChat: <span class="blue-underline">pat2win</span>, or scan the code below.', unsafe_allow_html=True)
    st.image(wechat_image)
    st.subheader("Enjoy Chatting!")
    st.sidebar.markdown('<span class="blue-underline">Life Enhancing with AI.</span>', unsafe_allow_html=True)
    with st.spinner("Preparing website materials for you..."):
        try:
            loader = WebBaseLoader(url)
            raw_text = loader.load()
            page_content = raw_text[0].page_content
            page_content = str(page_content)
            temp_texts = text_splitter.split_text(page_content)
            texts = temp_texts
            initial_embeddings=get_embeddings(texts)
            db_embeddings = torch.FloatTensor(initial_embeddings) 
        except Exception as e:
            st.write("Unknow error.")
            print("Please enter a valide URL.")
            st.stop()  
          
if user_question.strip().isspace() or user_question.isspace():
    st.write("Query Empty. Please enter a valid query first.")
    st.stop()
elif user_question == "exit":
    st.stop()
elif user_question == "":
    print("Query Empty. Please enter a valid query first.")
    st.stop()
elif user_question != "":     
    #st.write("Your query: "+user_question)
    print("Your query: "+user_question)
    
with st.spinner("AI Thinking...Please wait a while to Cheers!"):
    q_embedding=get_embeddings(user_question)
    final_q_embedding = torch.FloatTensor(q_embedding)  
    hits = semantic_search(final_q_embedding, db_embeddings, top_k=5)
    page_contents = []
    for i in range(len(hits[0])):
        page_content = texts[hits[0][i]['corpus_id']]
        page_contents.append(page_content)
    temp_page_contents=str(page_contents)
    final_page_contents = temp_page_contents.replace('\\n', '')     
    random_string = generate_random_string(20)
    i_file_path = random_string + ".txt"
    with open(i_file_path, "w", encoding="utf-8") as file:
        file.write(final_page_contents)
    loader = TextLoader(i_file_path, encoding="utf-8")
    loaded_documents = loader.load()
    temp_ai_response=chain({"input_documents": loaded_documents, "question": user_question}, return_only_outputs=False)
    temp_ai_response = temp_ai_response['output_text']    
    final_ai_response=temp_ai_response.partition('<|end|>')[0]
    i_final_ai_response = final_ai_response.replace('\n', '')
    st.write("AI Response:")
    st.write(i_final_ai_response)
