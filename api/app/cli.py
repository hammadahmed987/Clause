# from app.services.ingestion import ingest
from app.services.chat import RAGService


def main():
    rag_service = RAGService()
    try:
        query = input("You: ").strip()
        result = rag_service.ask(query)

        # if result.sources:
        #     pass

        print(f"AI: {result.answer}")



    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rag_service.close()


if __name__ == "__main__":
    # print(f"Ingested {ingest()} chunks!")
    main()







# Pinecone, Pgvector, Milvas, Qdrant, Weviate
# Chroma, Faiss

# Packages to Install:
# uv add langchain langchain-core langchain-text-splitters langchain-ollama pydantic pydantic-settings langchain-qdrant langchian-client langchain-community pypdf