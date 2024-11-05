import os
from typing import Optional

from docling.document_converter import DocumentConverter


class PdfToMarkdown:
    def __init__(self, pdf_path_or_url: str):
        self.pdf_path_or_url = pdf_path_or_url
        self.converter = DocumentConverter()

    def convert(self, file_name: Optional[str] = None) -> str:
        _file_name = file_name if file_name else self.pdf_path_or_url.split("/")[-1]
        _storage_path = f"storage/markdown/{_file_name}.md"

        # 既存のmarkdownファイルがあれば、それを読み込んで返す
        if os.path.exists(_storage_path):
            with open(_storage_path, "r") as f:
                return f.read()

        # 新規変換の場合
        result = self.converter.convert(self.pdf_path_or_url)
        markdown = result.document.export_to_markdown()

        # markdownを保存
        os.makedirs(os.path.dirname(_storage_path), exist_ok=True)
        with open(_storage_path, "w") as f:
            f.write(markdown)

        return markdown
