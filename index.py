import os

    # Be sure to have all the API keys in your local environment as shown below
    # Do not publish environment keys in production
    # os.environ["OPENAI_API_KEY"] = "sk"
    # os.environ["FIREWORKS_API_KEY"] = ""
    # os.environ["MONGO_URI"] = ""

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#MONGO_URI = os.environ.get("MONGO_URI")
MONGO_URI='mongodb+srv://Cluster76630:cGdpfFpFcE1o@cloud0.rnjufgy.mongodb.net/?retryWrites=true&w=majority&appName=cloud0'


#data ingestion into mongodb 
import pandas as pd
from datasets import load_dataset
from pymongo import MongoClient
from langchain_fireworks import Fireworks
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_fireworks import Fireworks, ChatFireworks
from langchain.agents import tool, Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent,tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 





data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings")
dataset_df = pd.DataFrame(data["train"])

print(len(dataset_df))
dataset_df.head()

# Initialize MongoDB python client
client = MongoClient('mongodb+srv://Cluster76630:cGdpfFpFcE1o@cloud0.rnjufgy.mongodb.net/?retryWrites=true&w=majority&appName=cloud0')

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]
# Delete any existing records in the collection

'''
# Delete any existing records in the collection
collection.delete_many({})

# Data Ingestion
records = dataset_df.to_dict('records')
collection.insert_many(records)

print("Data ingestion into MongoDB completed") 
''' 
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding= embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="abstract"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
llm = ChatFireworks(
    model="accounts/fireworks/models/firefunction-v1",
    max_tokens=256)

# Custom Tool Definiton
@tool
def get_metadata_information_from_arxiv(word: str) -> list:
  """
  Fetches and returns metadata for a maximum of ten documents from arXiv matching the given query word.

  Args:
    word (str): The search query to find relevant documents on arXiv.

  Returns:
    list: Metadata about the documents matching the query.
  """
  docs = ArxivLoader(query=word, load_max_docs=10).load()
  # Extract just the metadata from each document
  metadata_list = [doc.metadata for doc in docs]
  return metadata_list


@tool
def get_information_from_arxiv(word: str) -> list:
  """
  Fetches and returns metadata for a single research paper from arXiv matching the given query word, which is the ID of the paper, for example: 704.0001.

  Args:
    word (str): The search query to find the relevant paper on arXiv using the ID.

  Returns:
    list: Data about the paper matching the query.
  """
  doc = ArxivLoader(query=word, load_max_docs=1).load()
  return doc
 

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="This serves as the base knowledge source of the agent and contains some records of research papers from Arxiv. This tool is used as the first step for exploration and reseach efforts."
)

tools = [get_metadata_information_from_arxiv, get_information_from_arxiv, retriever_tool]
agent_purpose = "You are a helpful research assistant"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)
     

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name=DB_NAME, collection_name="history")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=get_session_history("my-session")
)

agent = create_tool_calling_agent(llm,tools, prompt,)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

agent_executor.invoke({"input": "Get me a list of research papers on the topic Prompt Compression"})
