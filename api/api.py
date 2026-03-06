from ninja import NinjaAPI, File, Schema
from ninja.files import UploadedFile
from django.conf import settings
import os
from typing import List
from .utils.etl import process_pdf
from .utils.agent import get_agent_executor
from .models import UploadedPDF, ChatMessageStored
from dotenv import load_dotenv

load_dotenv()

api = NinjaAPI()


class ChatMessage(Schema):
    role: str  # "human" or "ai"
    content: str


class ChatRequest(Schema):
    query: str
    chat_history: List[ChatMessage] = []


@api.post("/upload")
def upload_pdf(request, file: UploadedFile = File(...)):
    try:
        print(f"DEBUG Upload: Received file {file.name}, size: {file.size} bytes")
        
        # Save file to media
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        print(f"DEBUG Upload: Saving to {file_path}")
        
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Track in DB (upsert by filename)
        UploadedPDF.objects.update_or_create(
            file_name=file.name,
            defaults={
                'name': file.name,
                'file_size': file.size,
            }
        )

        # Process PDF
        print(f"DEBUG Upload: Starting PDF processing...")
        result = process_pdf(
            pdf_path=file_path,
            mongodb_uri=os.getenv("MONGODB_URI"),
            db_name=os.getenv("DB_NAME"),
            collection_name=os.getenv("COLLECTION_NAME"),
            index_name=os.getenv("VECTOR_INDEX_NAME")
        )
        print(f"DEBUG Upload: Processing complete - {result}")
        return {"message": "Success", "details": result}
    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG Upload Error: {error_msg}")
        return {
            "message": "Error",
            "details": f"Failed to process PDF: {error_msg}"
        }


@api.get("/pdfs")
def list_pdfs(request):
    """Return all uploaded PDFs ordered by most recent first."""
    pdfs = UploadedPDF.objects.all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "file_name": p.file_name,
            "file_size": p.file_size,
            "uploaded_at": p.uploaded_at.isoformat(),
            "url": f"{settings.MEDIA_URL}{p.file_name}",
        }
        for p in pdfs
    ]



@api.post("/chat")
def chat_agent(request, payload: ChatRequest):
    try:
        executor = get_agent_executor(
            mongodb_uri=os.getenv("MONGODB_URI"),
            db_name=os.getenv("DB_NAME"),
            collection_name=os.getenv("COLLECTION_NAME"),
            index_name=os.getenv("VECTOR_INDEX_NAME")
        )

        # 1. Store Human Question
        ChatMessageStored.objects.create(role="human", content=payload.query)

        # Get previous messages (excluding the one we just saved)
        past_messages = ChatMessageStored.objects.all().order_by('-timestamp')[1:6]
        # Reverse them to get chronological order
        past_messages = reversed(list(past_messages))

        # Format chat history as a readable string
        formatted_history = ""
        for msg in past_messages:
            label = "Human" if msg.role == "human" else "AI"
            formatted_history += f"{label}: {msg.content}\n"

        # Run agent with timeout protection
        response = executor.invoke({
            "input": payload.query,
            "chat_history": formatted_history.strip()
        }, {
            "max_execution_time": 180  # 3 minutes max
        })

        # 3. Store AI Answer
        ChatMessageStored.objects.create(role="ai", content=response["output"])

        # Format intermediate steps for the UI
        thought_steps = []
        for action, observation in response.get("intermediate_steps", []):
            thought_steps.append(f"Thought: {action.log}\nObservation: {observation}")

        thought_trace = "\n\n".join(thought_steps) if thought_steps else "Direct answer from LLM."

        return {
            "answer": response["output"],
            "thought_process": thought_trace
        }
    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG Chat Error: {error_msg}")
        # Return user-friendly error message
        return {
            "error": f"Agent processing failed: {error_msg}",
            "answer": "I encountered an error processing your request. Please try rephrasing your question or check if the knowledge base has relevant documents.",
            "thought_process": f"Error details: {error_msg}"
        }


@api.delete("/chat/clear")
def clear_chat_history(request):
    """Delete all stored chat messages from the database."""
    ChatMessageStored.objects.all().delete()
    return {"message": "Chat history cleared"}
