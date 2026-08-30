from pathlib import Path
import sys,json,faiss
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
docs=ROOT/"rag"/"documents"; out=ROOT/"rag"/"index"; out.mkdir(parents=True,exist_ok=True)
chunks=[]
for p in docs.glob("*.md"):
    for text in p.read_text(encoding="utf-8").split("\n\n"):
        if text.strip(): chunks.append({"title":p.stem,"text":text.strip()})
model=SentenceTransformer("all-MiniLM-L6-v2")
emb=model.encode([x["text"] for x in chunks],normalize_embeddings=True).astype("float32")
idx=faiss.IndexFlatIP(emb.shape[1]); idx.add(emb)
faiss.write_index(idx,str(out/"faiss.index"))
(out/"chunks.json").write_text(json.dumps(chunks,indent=2),encoding="utf-8")
print("Indexed",len(chunks),"policy chunks.")
