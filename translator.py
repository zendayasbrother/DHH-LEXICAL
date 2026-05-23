import os
import re
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datacleanse import DataCleaner
import epitran 

# Use epitran to solidify two languages - English and German

class Translator:
    def __init__(self):
        print("Loading translation model (this might take a moment)...")
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Mapping dictionary for NLLB specific language codes
        self.LANG_MAP = {
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
            "da": "dan_Latn",
            "zh": "zho_Hans",
            "auto": "eng_Latn" 
        }

    def detect_language(self, text):
        # Detects language with fallback to English for non-alpha strings.
        try:
            if not text or not any(char.isalpha() for char in text):
                return "en"
            return detect(text)
        except Exception:
            return "en" 

    def translate(self, text, source='auto', target='en'):
        # Translates text using NLLB mapping codes.
        src_lang_code = self.LANG_MAP.get(source, "eng_Latn") 
        tgt_lang_code = self.LANG_MAP.get(target, "eng_Latn") 

        # Prepare tokenizer configuration
        self.tokenizer.src_lang = src_lang_code
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get target language token ID
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)
        
        # Generate translation
        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id = tgt_lang_id, 
            max_length = 250
        )
        
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]



