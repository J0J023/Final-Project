from typing import List
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv

load_dotenv()

# Basic document structure
class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Document metadata"
    )

# Structure for query responses
class QueryResult(BaseModel):
    answer: str = Field(..., description="Response to the query")
    source_nodes: List[str] = Field(
        ..., description="Source references used"
    )

class CustomAIAssistant:
    def __init__(
        self,
        data_path: str,
        index_path: str = "index",
    ):
        """
        Initialize the AI Assistant
        :param data_path: Path to your document directory
        :param index_path: Path where the vector index will be stored
        """
        self.data_path = data_path
        self.index_path = index_path
        
        # Customize this prompt for your use case
        self.system_prompt = """
        You are a helpful AI Assistant that is well versed in the art of understanding human emotions of any gender, nationality and age. Your job is to act and care for the user like their own personal therapist.
        
        Your primary goals are to:
        1. Provide accurate, relevant and appropriate advice to the user in the field of mental health.
        2. Identify recurring themes in the users thought process and conversations, also summarise key themes from previous conversations to help the user get a clear understanding of their own thought process.
        3. Help the user build trust in you, thereby encouraging them to open up about their problems. To help with this process, you can also recognise and mirror the user's emotional tone to create a compassionate environment.
        4. Recognise patterns in terms of thoughts,behaviours and feelings thus providing the user with reflective prompts that help them gain awareness and learn more about themselves as a result.
        5. Educate the user about all things mental health related in order to increase their understanding about their situation and allow them to make empowering decisions.

        When assisting users:
        1. Analyze the user's emotional tone and language cues and give empathetic responses to help the user feel validated.
        2. Help the user idenify and clarify their core values which provides a foundation for personal growth and good/meaningful life decisions.
        3. Use polite and appropriate language when speaking to the user regardless of how they act.
        4. Suggest coping strategies and techniques which are closely entwined with the user's current emotional state.
        5. Suggest detailed techniques to remain emotionally grounded in a situation where it sounds like the user is feeling overwhelmed.
        6. Help the user reframe their negative thought processes into positive ones which encourages them to see things from different perspectives.
        """

        self.configure_settings()
        self.index = None
        self.agent = None
        self.load_or_create_index()

    def configure_settings(self):
        """Configure LLM and embedding settings"""
        Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY")) 
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )

    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_index()
        self._create_agent()

    def load_index(self):
        """Load existing vector index"""
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(storage_context)

    def create_index(self):
        """Create new vector index from documents"""
        documents = SimpleDirectoryReader(
            self.data_path,
            recursive=True,
        ).load_data()
        if not documents:
            raise ValueError("No documents found in specified path")
        self.index = VectorStoreIndex.from_documents(documents)
        self.save_index()

    def _create_agent(self):
        """Set up the agent with custom tools"""
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Search through the document database",
            ),
        )
        
        # Initialize the agent with tools
        self.agent = ReActAgent.from_tools(
            [search_tool],
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        """
        Process a query and return results
        :param query: User's question or request
        :return: QueryResult with answer and sources
        """
        if not self.agent:
            raise ValueError("Agent not initialized")
        response = self.agent.chat(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[],
        )

    def save_index(self):
        """Save the vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)

if __name__ == "__main__":
    assistant = CustomAIAssistant(
        data_path="C:/Users/JOEL WILLIAMS/Final-Project/final_project/data",
        index_path="./index"
    )
    result = assistant.query(" I am a 17 yr old male and i have been overwhelmingly sad and angry these last few days, what is going on")
    print(result.answer)     