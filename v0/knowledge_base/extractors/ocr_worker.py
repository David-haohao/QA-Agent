"""独立进程执行单份扫描 PDF 的本地 OCR。"""

import json
import os
import sys


def load_pages(cache_path: str):
    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            return {int(page_num): str(page_text) for page_num, page_text in json.load(cache_file).get("pages", [])}
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}


def save_pages(cache_path: str, pages, completed: bool = False) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = cache_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as cache_file:
        json.dump(
            {"pages": sorted(pages.items()), "completed": completed},
            cache_file,
            ensure_ascii=False,
        )
    os.replace(temp_path, cache_path)


def extract_pages(file_path: str, cache_path: str):
    import pymupdf
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    extracted_pages = load_pages(cache_path)
    with pymupdf.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            if page_num in extracted_pages:
                continue
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            result, _ = engine(image)
            lines = [item[1] for item in (result or []) if len(item) > 1 and item[1].strip()]
            page_text = "\n".join(lines).strip()
            if page_text:
                extracted_pages[page_num] = page_text
            save_pages(cache_path, extracted_pages)
            del pixmap, image
    save_pages(cache_path, extracted_pages, completed=True)
    return sorted(extracted_pages.items())


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m knowledge_base.extractors.ocr_worker <pdf-path> <cache-path>")
    extract_pages(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
