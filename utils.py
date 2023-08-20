from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model_name="C:/Users/jayyd/Documents/projects/all-MiniLM-L6-v2")
    return embeddings


# Function to push embedding into faiss
def push_to_faiss(documents, embedding):
    db = FAISS.from_documents(documents, embedding)
    db.save_local("faiss")

# Functio to pull information from FAISS
def pull_from_faiss(embeddings):
    db = FAISS.load_local("faiss", embeddings)
    return db


#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,embeddings,unique_id):

    """pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )"""

    #index_name = pinecone_index_name

    #index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    index = pull_from_faiss(embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("C:/Users/jayyd/Documents/projects/Llama-2-13b-chat-hf/")
    pipeline_llm = pipeline("text-generation",
                        model = "C:/Users/jayyd/Documents/projects/Llama-2-13b-chat-hf/",
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto",
                        max_length = 4000,
                        do_sample = True,
                        top_k = 10,
                        eos_token_id = tokenizer.eos_token_id
                        )
    return pipeline_llm



# Helps us get the summary of a document
def get_summary(current_doc):
    llm = HuggingFacePipeline(pipeline = load_llm(), model_kwargs = {'temperature': 0})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary




    