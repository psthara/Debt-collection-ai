# Debt Collection AI

Complete reference project combining:
- ML payment prediction
- ML PTP prediction
- ML recovery prediction
- collection priority scoring
- next-best-action rules
- RAG over collection policies
- optional Ollama GenAI
- FastAPI APIs

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/train.py
python scripts/build_rag.py
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

Optional local LLM:
```bash
ollama pull llama3.2
```

The project works without Ollama; RAG still retrieves policy passages.

This is a reference implementation. Production debt collection requires approved policies, privacy/security controls, audit logging, human oversight and fairness/compliance validation.
