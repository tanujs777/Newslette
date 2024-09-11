import validators
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "your-hf-token")

# Streamlit configuration
st.set_page_config(page_title="Product Page Summarizer")
st.title("Product Page Summarizer")
st.subheader("Summarize the key details of a product page")

# Sidebar input
with st.sidebar:
    web_url = st.text_input("Enter Product Page URL")

# Function to scrape product details
def get_product_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Attempt to find the product name more robustly
    product_name = soup.find('h1')  # Simplified to catch any H1 tag
    if not product_name:
        product_name = soup.find('h1', class_=True)  # Try finding an H1 with any class
    product_name = product_name.text.strip() if product_name else "Product Name not available"
    
    ori_price = soup.find('div', {'class': 'discount-percent css-zizo20'})
    ori_price = ori_price.text.strip() if ori_price else "Original Price not available"

    active_price = soup.find('div', {'class': 'pdp-active-price css-1suxpbj'})
    active_price = active_price.text.strip() if active_price else "Active Price not available"

    discount_percentage = soup.find('div', {'class': 'pdp-price-discount-range-wrapper css-nwvy60'})
    discount_percentage = discount_percentage.text.strip('()') if discount_percentage else "Discount Percentage not available"

    color_elements = soup.find_all(attrs={'data-color-text': True})
    colors = [element['data-color-text'] for element in color_elements] if color_elements else "Colors not available"

    product_details = soup.find('div', {'class': 'css-uaknjp'})
    product_details = product_details.text.strip() if product_details else "Product Details not available"

    div_tags_with_src = soup.find_all('img', {"src": True})
    image_sources = [tag['src'] for tag in div_tags_with_src] if div_tags_with_src else "Images not available"

    return {
        'product_name': product_name,
        'ori_price': ori_price,
        'active_price': active_price,
        'discount_percentage': discount_percentage,
        'colors': colors,
        'product_details': product_details,
        'image_sources': image_sources
    }

# Initialize LLM and other components
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# System prompt template for the LLM
system_template = (
    "You are a product description summarizer. Your task is to read the product page content "
    "and provide a concise summary that includes the following key details:\n"
    "1. Product Name\n"
    "2. Key Features\n"
    "3. Price\n"
    "4. Customer Reviews\n\n"
    "{context}"
)

# Query to pass to the LLM
chat_query = "Please summarize the product page."

# Button to trigger summarization
if st.button("Summarize"):
    try:
        if not web_url:
            st.error("Please provide the Product Page URL to get started.")
        elif not validators.url(web_url):
            st.error("Please enter a valid URL.")
        else:
        
            product_details = get_product_details(web_url)
            
            # Prepare content for LLM
            product_details_for_llm = {
                key: value for key, value in product_details.items() if value and "not available" not in value
            }

            product_details_text = "\n".join([f"{key}: {value}" for key, value in product_details_for_llm.items()])
            
            # Load and process the product page content
            loader = WebBaseLoader(web_path=[web_url])
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=200)
            final_docs = text_splitter.split_documents(docs)
            
            vector_store_db = FAISS.from_documents(final_docs, embeddings)
            retriever = vector_store_db.as_retriever()
      
            prompt = PromptTemplate(input_variables=["context"], template=system_template)
            summary_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, summary_chain)
            
            # Generate summary using LLM
            response = rag_chain.invoke({"input": chat_query, "context": product_details_text})
            
            st.success("Summary Generated:")
            st.text(response["answer"])  
            
            st.write("Scraped Product Details:")
            st.json(product_details)
    except Exception as e:
        st.exception(f'Exception: {e}')
