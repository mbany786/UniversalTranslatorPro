import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from PIL import Image
import openai
import gc
import time
import numpy as np
from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from textwrap import wrap
import easyocr
import json

# ---------------------------
# ===== CONFIGURATION =====
# ---------------------------

PDF_DPI = 150
CHUNK_SIZE = 3  # number of pages to translate together

# ---------------------------
# ===== OPENAI API KEY =====
# ---------------------------

# Paste your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY") # <-- Replace with your key

if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
    raise ValueError("OpenAI API key not set. Please paste your API key in the script.")

# ---------------------------
# ===== OCR SETUP =====
# ---------------------------

# EasyOCR reader for Urdu, Arabic, English
ocr_reader = easyocr.Reader(['ur', 'ar', 'en'], gpu=False, verbose=False)

# ---------------------------
# ===== HELPER FUNCTIONS =====
# ---------------------------

def ask_for_input():
    return input("Enter PDF file path or webpage URL: ").strip()

def ask_for_output_folder():
    default_folder = Path(__file__).parent / "translated_output"
    folder = input(f"Enter output folder (default: {default_folder}): ").strip()
    if not folder:
        folder = default_folder
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def parse_page_selection(selection: str, total_pages: int) -> list[int]:
    """
    Convert a string like '1,3,5-7' or 'all' into a list of page numbers.
    Pages are 1-indexed.
    """
    if selection.lower() == "all":
        return list(range(1, total_pages + 1))

    pages = set()
    parts = selection.split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            start, end = int(start), int(end)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted([p for p in pages if 1 <= p <= total_pages])

# ---------------------------
# ===== PDF FUNCTIONS =====
# ---------------------------

def pdf_to_images_generator(pdf_path: Path):
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(str(pdf_path))
    total_pages = info["Pages"]
    for i in range(1, total_pages + 1):
        img = convert_from_path(str(pdf_path), dpi=PDF_DPI, first_page=i, last_page=i)[0]
        yield img

def ocr_image_to_text(image: Image.Image) -> str:
    """Extract text from image using EasyOCR."""
    results = ocr_reader.readtext(np.array(image), detail=0)
    return "\n".join(results)

# ---------------------------
# ===== TRANSLATION FUNCTIONS =====
# ---------------------------

def translate_text_to_english(text: str) -> str:
    """Translate text into natural, context-aware English with attention to spiritual/philosophical terminology."""
    if not text.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator with expertise in classical Arabic, "
                "Islamic theology, and Sufi metaphysical terminology. Your task is to translate "
                "the user's text into fluent, natural English.\n\n"
                "Guidelines:\n"
                "- Prioritize meaning over word-for-word literalism.\n"
                "- When the text is spiritual, philosophical, or mystical, use established "
                "English terminology where appropriate (e.g., 'divine manifestation', "
                "'annihilation of the self', 'spiritual unveiling').\n"
                "- Preserve the tone and register of the original (scholarly, poetic, devotional, etc.).\n"
                "- Do NOT include transliterations unless absolutely necessary for meaning.\n"
                "- Do NOT include the original language in the output.\n"
                "- The result should read like a polished translation found in an academic or "
                "well-edited spiritual text."
            ),
        },
        {"role": "user", "content": text},
    ]

    response = openai.responses.create(
        model="gpt-4.1-mini",
        input=messages
    )

    return response.output_text.strip() if response.output_text else ""

def translate_pdf(pdf_path: str, output_folder: Path):
    pdf_path = Path(pdf_path)
    base_name = pdf_path.stem
    all_text = []
    progress_file = output_folder / f"{base_name}_progress.json"

    # Load progress if exists
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        all_text = progress_data.get("translated_pages", [])
        last_translated = progress_data.get("last_page", 0)
        print(f"Resuming from page {last_translated + 1}...")
    else:
        last_translated = 0

    # Determine total pages
    total_pages = sum(1 for _ in pdf_to_images_generator(pdf_path))
    print(f"Total pages: {total_pages}")

    # Ask user which pages to translate
    page_input = input(
        f"Enter pages to translate (e.g., 1,3,5-7) or 'all' for entire PDF [all]: "
    ).strip()
    if not page_input:
        page_input = "all"

    selected_pages = parse_page_selection(page_input, total_pages)
    print(f"Selected pages: {selected_pages}")

    chunk_text = []
    chunk_page_numbers = []

    for i, img in enumerate(pdf_to_images_generator(pdf_path), start=1):
        if i <= last_translated:
            continue  # skip pages already translated

        if i not in selected_pages:
            print(f"Skipping page {i} (not selected)")
            all_text.append("")  # placeholder
            continue

        print(f"Processing page {i} via OCR...")
        page_text = " ".join(ocr_image_to_text(img).splitlines())
        if not page_text.strip():
            print(f"Page {i} empty after OCR, skipping...")
            all_text.append("")  # placeholder
        else:
            chunk_text.append(page_text)
            chunk_page_numbers.append(i)

        # Translate in chunks
        if len(chunk_text) >= CHUNK_SIZE or (i == max(selected_pages) and chunk_text):
            combined_text = " ".join(chunk_text)
            print(f"Translating pages {chunk_page_numbers[0]}-{chunk_page_numbers[-1]}...")
            translated_chunk = translate_text_to_english(combined_text)
            all_text.append(translated_chunk)
            chunk_text = []
            chunk_page_numbers = []

        # Save progress
        progress_data = {"last_page": i, "translated_pages": all_text}
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

        del img
        gc.collect()
        time.sleep(0.2)

    # Combine all translated chunks
    full_text = "\n\n".join(all_text)

    # Save final outputs
    save_txt(full_text, output_folder, base_name)
    save_docx(full_text, output_folder, base_name)
    save_pdf(full_text, output_folder, base_name)

    # Remove progress file
    if progress_file.exists():
        os.remove(progress_file)
        print("Progress file removed, translation complete.")

# ---------------------------
# ===== SAVE FUNCTIONS =====
# ---------------------------

def save_txt(text: str, folder: Path, base_name: str):
    path = folder / f"{base_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ TXT saved to {path}")

def save_docx(text: str, folder: Path, base_name: str):
    path = folder / f"{base_name}.docx"
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)
    print(f"✅ DOCX saved to {path}")

def save_pdf(text: str, folder: Path, base_name: str):
    path = folder / f"{base_name}.pdf"
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    margin = 50
    line_height = 14
    y = height - margin
    for para in text.split("\n\n"):
        lines = wrap(para, width=90)
        for line in lines:
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height
        y -= line_height
    c.save()
    print(f"✅ PDF saved to {path}")

# ---------------------------
# ===== WEBPAGE FUNCTIONS =====
# ---------------------------

def extract_text_from_url(url: str) -> str:
    print(f"Downloading webpage: {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text() for p in paragraphs)

def translate_webpage(url: str, output_folder: Path):
    raw_text = extract_text_from_url(url)
    if not raw_text.strip():
        print("No text found on the webpage.")
        return
    print("Translating webpage text to English...")
    merged_text = " ".join(raw_text.splitlines())
    translated_text = translate_text_to_english(merged_text)
    base_name = "_".join(url.replace("https://", "").replace("http://", "").split("/"))
    save_txt(translated_text, output_folder, base_name)
    save_docx(translated_text, output_folder, base_name)
    save_pdf(translated_text, output_folder, base_name)

# ---------------------------
# ===== MAIN FUNCTION =====
# ---------------------------

def main():
    user_input = ask_for_input()
    output_folder = ask_for_output_folder()

    if user_input.startswith("http://") or user_input.startswith("https://"):
        translate_webpage(user_input, output_folder)
    elif Path(user_input).exists() and user_input.lower().endswith(".pdf"):
        translate_pdf(user_input, output_folder)
    else:
        print("Invalid input. Please enter a valid webpage URL or local PDF path.")

if __name__ == "__main__":
    main()
