from services.ingestion import ingest
from app.services.chat import RAGService

# print(f"Ingested {ingest()} chunks")

def main() -> None:
    rag_service = RAGService()
    try:
        query = input("You: ").strip()
        result = rag_service.ask(question=query)
        print(f"AI: {result.answer}")
    except ( KeyboardInterrupt, EOFError):
        pass
    finally:
        rag_service.close()

if __name__ == "__main__":
    # print(f"Ingested {ingest()} chunks")
    main()