import chromadb
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.agent.openai import OpenAIAgent

load_dotenv()

embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model


def load_tool():
    db2 = chromadb.PersistentClient(path="./chroma_db")

    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    def answer_question(question: str):
        """
        Answer question about public administrative services
        """
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        return response

    return FunctionTool.from_defaults(
        fn=answer_question,
        description="Answer question about public administrative services",
    )


llm = OpenAI(
    model="gpt-4o-mini",
)

system_prompt = """Bạn là trợ lý giúp trả lời các câu hỏi về dịch vụ hành chính công. Hãy tỏ ra thân thiện với người dùng nhé."""

agent = OpenAIAgent.from_tools(
    tools=[load_tool()],
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
)


while True:
    question = input("\nBạn có câu hỏi gì không? ")
    if question == "exit":
        break
    response = agent.stream_chat(question)
    response.print_response_stream()
