import json, requests, faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import RAG_DIR, OLLAMA_URL, OLLAMA_MODEL

class RAGService:
    def __init__(self):
        self.index = faiss.read_index(str(RAG_DIR/"faiss.index"))
        self.chunks = json.loads((RAG_DIR/"chunks.json").read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, question, k=4):
        q = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        out=[]
        for score, idx in zip(scores[0],ids[0]):
            if idx >= 0:
                x=dict(self.chunks[int(idx)])
                x["score"]=round(float(score),4)
                out.append(x)
        return out

    def answer(self, question):
        docs=self.retrieve(question)
        context="\n\n".join(f"[{x['title']}]\n{x['text']}" for x in docs)
        prompt=f"""Answer the question only from the policy context.
Do not invent policy. If the context is insufficient, say a human policy
review is required.

Question: {question}

Policy context:
{context}"""
        try:
            r=requests.post(f"{OLLAMA_URL}/api/generate",
                json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},timeout=30)
            r.raise_for_status()
            ans=r.json().get("response","").strip()
            if ans:
                return {"answer":ans,"sources":docs,"llm":OLLAMA_MODEL}
        except Exception:
            pass
        return {"answer":"LLM unavailable. Review the retrieved policy passages before acting.",
                "sources":docs,"llm":None}
