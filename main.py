import os
import re
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datacleanse import DataCleaner
from translator import Translator


if __name__ == "__main__":
    translator = Translator()
    
    print("\n" + "=" * 40)
    enquiry_text = input("Enter your text: ").strip()
    target_pref = input("Enter desired target language code (e.g., 'en', 'fr', 'es'): ").strip().lower()
    print("=" * 40)

    # Core Execution
    detected_lang = translator.detect_language(enquiry_text)
    translated_enq = translator.translate(enquiry_text, source=detected_lang, target=target_pref)
    
    # Outputs
    print(f"\nOriginal Enquiry: {enquiry_text}")
    print(f"Detected Language: {detected_lang}")
    print(f"Final Response ({target_pref}): {translated_enq}")