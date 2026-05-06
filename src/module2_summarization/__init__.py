"""
Module 2: Speech-to-Text & Content Summarization
Owner: Ravindu (214095L)

Transcribes lecture audio using OpenAI Whisper (large-v3) and classifies
sentences as important/not-important using fine-tuned BERT-base.

Classification Criteria:
    1. Concept introduction
    2. Term definition
    3. Key explanation

Technologies: OpenAI Whisper, BERT-base, Hugging Face Trainer
References: Radford et al. 2023, Gonzalez et al. 2023
"""

__version__ = "0.1.0"
__author__ = "Ravindu (214095L)"
