import gradio as gr
import tempfile
import fitz  # PyMuPDF for PDF text extraction
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import html

# Load Hugging Face token securely
# os.environ["HF_TOKEN"] = "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"
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# if not hf_token:
#     raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable.")

hf_token = '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'
model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf", 
    local_dir="/content/models/llama3",       #change the model path of your own folder
    local_dir_use_symlinks=False
)

def load_model(model_path):
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=False,
            verbose=True
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")


def extract_json(text):
    text = html.unescape(text)
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    try:
        return json.loads(text)
    except Exception as e:
        return {"Error": f"Failed to extract JSON: {str(e)}"}


def build_prompt(text):
    prompt = """You are an information extraction engine. Your task is to extract clean, structured JSON metadata from scientific documents.

 Do not repeat keys. Use each key only once.
 Return valid JSON only. No duplicates or suffixes like "URL".
 Ensure "Keywords" is a list of strings.

Only return the JSON, nothing else.
{
  "Title": "Paper title",
  "Authors": ["Author 1", "Author 2"],
  "DOI": "DOI if available",
  "Keywords": ["Keyword1", "Keyword2"],
  "Abstract": "Abstract text",
  "Document Type": "Research Paper, Thesis, etc.",
  "Number of References": 10
}

Only return valid JSON. Do not include explanations or markdown.

Extract metadata from the following paper:
"""
    paper_excerpt = text[:3000]

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant that extracts structured metadata from scientific papers."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt.strip()}\n\n{paper_excerpt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_metadata(generator, paper_text):
    prompt = build_prompt(paper_text)
    response = generator.create_completion(
        prompt,
        max_tokens=2048,  # Increase as needed
        temperature=0,
        top_p=0.9,
        stop=["<|user|>", "</s>"]  # Add stop token to clean output
    )
    raw_output = response["choices"][0]["text"]
    return extract_json(raw_output)

# Extract raw text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text if text.strip() else "Error: No extractable text found in PDF."

def process_pdf(pdf_file):
    extracted_text = extract_text_from_pdf(pdf_file.name)
    if extracted_text.startswith("Error:"):
        return {"Error": "No extractable text found in the PDF."}
    metadata = extract_metadata(model, extracted_text)
    return metadata

def main():
    model_path = "/content/models/llama3/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    global model
    model = load_model(model_path)
    #Gradio interface
    iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF"),
    outputs="json",
    title="Metadata Extractor",
    description="Upload only a PDF to extract metadata"
    )
    # Launch the interface
    iface.launch()

if __name__ == "__main__":
    main()