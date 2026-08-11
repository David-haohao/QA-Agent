# ============================================================
# 本地 Embedding / Reranker 服务（替代银行内网 172.18.1.22:4000）
#
# 作用：在你的个人电脑上暴露与原有内网服务完全一致的
#       OpenAI 兼容接口，使问答 Agent 无需访问银行内网即可检索。
#   - POST /v1/embeddings  -> BGE-M3 向量化
#   - POST /v1/rerank      -> BGE-Reranker-v2-m3 精排
#
# 依赖：pip install FlagEmbedding fastapi uvicorn
# 模型：默认从本地 ./local_models/ 读取（已离线下载到项目内，无需联网）。
#       如需在线下载，可设 EMBED_MODEL / RERANK_MODEL 为 HF 模型名，并设
#       HF_ENDPOINT=https://hf-mirror.com
#
# 启动：python local_embedding_server.py
#       默认监听 0.0.0.0:4000
# ============================================================
import os
import argparse

import numpy as np
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="Local BGE Embedding & Reranker Server")

# 默认读本地离线模型（项目内 local_models/），避免银行网络无法访问 HuggingFace
EMBED_MODEL = os.environ.get("EMBED_MODEL", "local_models/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "local_models/bge-reranker-v2-m3")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


DEVICE = os.environ.get("EMBED_DEVICE", "cuda" if _has_cuda() else "cpu")  # 自动选 GPU/CPU

_embed_model = None
_rerank_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        print(f"[加载] 嵌入模型 {EMBED_MODEL} -> device={DEVICE}")
        _embed_model = BGEM3FlagModel(EMBED_MODEL, use_fp16=False, device=DEVICE)
    return _embed_model


def get_rerank_model():
    # 注意：FlagEmbedding 1.4.0 的 FlagReranker 调用了 transformers 已移除的
    # tokenizer.prepare_for_model()，与新版 transformers(>=4.41) 不兼容。
    # 因此这里直接用 transformers 加载 bge-reranker-v2-m3（标准 cross-encoder）。
    global _rerank_model
    if _rerank_model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[加载] 重排模型 {RERANK_MODEL} -> device={DEVICE}")
        tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
        tmodel = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        tmodel = tmodel.to(DEVICE)
        tmodel.eval()
        _rerank_model = (tmodel, tok)
    return _rerank_model


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    model = body.get("model", EMBED_MODEL)
    inp = body.get("input", [])
    if isinstance(inp, str):
        inp = [inp]
    texts = [str(x)[:512] for x in inp]  # 与客户端一致：截断至 512 字符

    m = get_embed_model()
    out = m.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    dense = out["dense_vecs"]  # list[np.ndarray], dim=1024
    data = [{"object": "embedding", "index": i, "embedding": np.asarray(v).tolist()}
            for i, v in enumerate(dense)]
    return {"object": "list", "data": data, "model": model}


@app.post("/v1/rerank")
async def rerank(req: Request):
    import torch
    body = await req.json()
    model = body.get("model", RERANK_MODEL)
    query = body.get("query", "")
    documents = body.get("documents", [])
    if not documents:
        return {"results": [], "model": model}

    max_doc_len = max(512 - len(query), 1)
    docs = [str(d)[:max_doc_len] for d in documents]
    pairs = [[query, d] for d in docs]

    tmodel, tok = get_rerank_model()
    scores = []
    batch = 16
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            grp = pairs[i:i + batch]
            enc = tok(grp, padding=True, truncation=True,
                      return_tensors="pt", max_length=512).to(DEVICE)
            logits = tmodel(**enc).logits.squeeze(-1)
            sc = torch.sigmoid(logits).float().cpu().tolist()
            scores.extend(sc)
    results = [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]
    return {"results": results, "model": model}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
