# AI-Essay-Refiner
Developed a beginner-friendly tool that refines grammar in essays by processing .pdf, .docx, or .txt files. The system uses basic AI and NLP techniques to analyse the input text and generate a grammatically improved version, enhancing clarity and correctness.
# Install required libraries
!pip install transformers sentencepiece torch python-docx PyMuPDF

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word files
from google.colab import files

# Load the pre-trained T5 model (small model to fit in Colab)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def correct_text(text):
    input_text = "grammar: " + text  # Prefix for grammar correction
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Upload file
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
text = ""

# Extract text based on file type
if file_name.endswith(".pdf"):
    text = extract_text_from_pdf(file_name)
elif file_name.endswith(".docx"):
    text = extract_text_from_docx(file_name)
elif file_name.endswith(".txt"):
    text = extract_text_from_txt(file_name)
else:
    with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

# Correct text using AI model
if text:
    corrected_text = correct_text(text)
    print("\nCorrected Essay:\n")
    print(corrected_text)

    # Save corrected text to a new file
    with open("corrected_essay.txt", "w", encoding="utf-8") as f:
        f.write(corrected_text)
    print("\nCorrected text saved as 'corrected_essay.txt'.")
