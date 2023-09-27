from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# Load the API key
load_dotenv()
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Setup the repository and model api
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                    repo_id=repo_id,
                    model_kwargs={"temperature":0.2,"max_new_tokens":512}
                    )

# Setup the prompt template 
# template = """
# You are an artifial intelligence assistant. The assistant gives helpful, detailed response to the user's question.

# {user_prompt}
# """
# prompt = PromptTemplate(template=template, input_variables=["user_prompt"])
# llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# # Enter the prompt and run the model through langchain
# inference_prompt = "How many stars are there in MilkyWay Galaxy?"
# print(llm_chain.run(inference_prompt))


def load_video_data_vector(text):
    embeddings = HuggingFaceEmbeddings()
    loader =  YoutubeLoader.from_youtube_url(text)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat_model = llm

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question. Do not use any other data except the above transcript.
        Do not use your training data to respond. 
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat_model, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs




