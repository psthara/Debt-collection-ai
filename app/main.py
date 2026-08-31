from fastapi import FastAPI, HTTPException
from .schemas import Customer, RAGQuery, MessageRequest
from .ml_service import MLService
from .decision import next_best_action
from .rag_service import RAGService
from .genai import generate_message

app=FastAPI(title="Debt Collection AI",version="1.0")
try:
    ml=MLService()
    ML_ERROR=None
except Exception as e:
    ml=None; ML_ERROR=str(e)

try:
    rag=RAGService()
    RAG_ERROR=None
except Exception as e:
    rag=None; RAG_ERROR=str(e)

@app.get("/")
def root():
    return {"service":"Debt Collection AI","docs":"/docs"}

@app.get("/health")
def health():
    return {"api":"ok","ml":ml is not None,"rag":rag is not None}

@app.post("/predict")
def predict(c:Customer):
    if ml is None: raise HTTPException(503,ML_ERROR)
    scores=ml.predict(c)
    return {"customer_id":c.customer_id,**scores,"next_best_action":next_best_action(c,scores)}

@app.post("/rag/query")
def query(q:RAGQuery):
    if rag is None: raise HTTPException(503,RAG_ERROR)
    return rag.answer(q.question)

@app.post("/generate-message")
def message(req: MessageRequest):

    if ml is None:
        raise HTTPException(
            status_code=503,
            detail=f"ML service unavailable: {ML_ERROR}"
        )

    try:
        scores = ml.predict(req.customer)

        action = next_best_action(
            req.customer,
            scores
        )

        message_text = generate_message(
            req.customer,
            action,
            req.tone
        )

        return {
            "customer_id": req.customer.customer_id,
            "customer_name": req.customer.customer_name,
            "action": action,
            "message_source": "gemini_llm",
            "message": message_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM generation failed: {str(e)}"
        )
