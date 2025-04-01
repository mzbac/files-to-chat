import os
import re
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.exceptions import ConversionError 

def convert_files_in_folder(
    folder_path, ignore_pattern="", include_hidden=False, extensions=[]
):
    results = []
    for root, _, files in os.walk(folder_path):
        if extensions:
            files = [f for f in files if f.endswith(tuple(extensions))]
        for filename in files:
            filepath = os.path.abspath(os.path.join(root, filename))
            if ignore_pattern and re.search(ignore_pattern, filepath):
                continue
            if not include_hidden and filename.startswith("."):
                continue
            results.append(convert_file_to_markdown(filepath))
  
    return results


def convert_file_to_markdown(filepath):
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
            ),
        },
    )
    
    try:
        conv_result = doc_converter.convert(filepath)
        content = conv_result.document.export_to_markdown()
        return {"path": filepath, "content": content}
    except ConversionError as e:
        print(f"[INFO] Falling back to plain text for unsupported format: {filepath}")
        with open(filepath, "r") as f:
            content = f.read()
        return {"path": filepath, "content": content}


def print_as_xml(path, content):
    return f'  <document path="{path}">\n<source>{path}</source>\n<document_content>\n{content}\n</document_content>\n</document>'