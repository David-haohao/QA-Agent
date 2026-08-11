"""Manual end-to-end smoke test for a running local QA web service."""

import sys

import requests


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    response = requests.post(
        "http://127.0.0.1:8080/api/chat",
        data={
            "query": "根据知识库，温州银行2023年6月末资本充足率是多少？请只用一句中文回答，并写出来源文档名称。",
            "session_id": "codex-full-qa",
        },
        timeout=600,
    )
    response.raise_for_status()
    print(response.text)


if __name__ == "__main__":
    main()
