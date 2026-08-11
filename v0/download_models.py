# 把 BGE 模型权重下载到本地 local_models/ 目录。
# 这样用户在自己电脑（无外网/被银行网络限制）跑服务时直接读本地，无需联网。
#
# 下载源优先级：
#   1) ModelScope（阿里云，与公司网络放行 DashScope 同源，最可能在银行内网可达）
#   2) HuggingFace 镜像 hf-mirror.com（兜底）
#
# 用法：
#   # 默认：优先 ModelScope，失败回退 HF 镜像
#   .\.venv\Scripts\python.exe download_models.py
#   # 强制只用 HF 镜像（你机器能通 hf-mirror 时）
#   $env:HF_ENDPOINT="https://hf-mirror.com"; .\.venv\Scripts\python.exe download_models.py
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models")
MODELS = {
    "BAAI/bge-m3": os.path.join(BASE, "bge-m3"),
    "BAAI/bge-reranker-v2-m3": os.path.join(BASE, "bge-reranker-v2-m3"),
}
# The application loads the PyTorch/Transformers artifacts directly.  Keep the
# offline bundle small by excluding unused ONNX exports and repository images.
MODEL_FILE_PATTERNS = ["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.pt"]
MODEL_FILES = {
    "BAAI/bge-m3": [
        "config.json",
        "config_sentence_transformers.json",
        "configuration.json",
        "modules.json",
        "sentence_bert_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "colbert_linear.pt",
        "sparse_linear.pt",
        "pytorch_model.bin",
    ],
    "BAAI/bge-reranker-v2-m3": [
        "config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ],
}
DOWNLOAD_RETRIES = 30


def try_modelscope(repo_id, out):
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl
    except Exception as e:
        print(f"  [跳过] ModelScope 不可用: {e}", flush=True)
        return False
    print(f"  [ModelScope] 下载 {repo_id} ...", flush=True)
    ms_dl(
        model_id=repo_id,
        local_dir=out,
        revision="master",
        allow_patterns=MODEL_FILE_PATTERNS,
    )
    return True


def _download_file(url, destination):
    """Download one artifact with a resumable temporary file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    partial_path = destination.with_suffix(destination.suffix + ".part")
    existing_size = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    with requests.get(url, headers=headers, stream=True, timeout=30) as response:
        response.raise_for_status()
        append = existing_size > 0 and response.status_code == 206
        with partial_path.open("ab" if append else "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)
    os.replace(partial_path, destination)


def try_huggingface(repo_id, out):
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"  [HuggingFace] 下载 {repo_id} (endpoint={endpoint}) ...", flush=True)
    for relative_path in MODEL_FILES[repo_id]:
        file_url = f"{endpoint.rstrip('/')}/{repo_id}/resolve/main/{quote(relative_path)}"
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                _download_file(file_url, Path(out) / relative_path)
                break
            except (requests.RequestException, OSError) as error:
                if attempt == DOWNLOAD_RETRIES:
                    raise
                print(
                    f"  重试 {relative_path} ({attempt}/{DOWNLOAD_RETRIES - 1}): {error}",
                    flush=True,
                )
                time.sleep(attempt)
    return True


def main():
    force_hf = "--hf" in sys.argv
    for repo_id, out in MODELS.items():
        os.makedirs(out, exist_ok=True)
        print(f"\n[开始] {repo_id} -> {out}", flush=True)
        t0 = time.time()
        ok = False
        if not force_hf:
            try:
                ok = try_modelscope(repo_id, out)
            except Exception as e:
                print(f"  ModelScope 失败: {e}", flush=True)
        if not ok:
            try:
                ok = try_huggingface(repo_id, out)
            except Exception as e:
                print(f"  HuggingFace 失败: {e}", flush=True)
        if ok:
            print(f"[完成] {repo_id} 用时 {int(time.time() - t0)}s", flush=True)
        else:
            print(f"[失败] {repo_id} 两个源都未成功，请检查网络或手动放置模型", flush=True)
    print("\nALL_DONE")


if __name__ == "__main__":
    main()
