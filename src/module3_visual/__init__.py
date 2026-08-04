"""
Module 3: Visual Content Understanding & Slide Extraction

Classifies extracted slide frames as Critical/Important/Skip using ViT-base.
Extracts on-screen text from Critical and Important frames using TrOCR.

Classification Labels:
    1. Critical  — Key diagrams, important formulas, summary slides
    2. Important — Definitions, step explanations, worked examples
    3. Skip      — Title slides, duplicates, blank frames

Technologies: ViT-base, TrOCR, OpenCV, Hugging Face
References: Biswas et al. 2025, Li et al. 2023
"""

__version__ = "0.1.0"
__author__ = "Fazly (214008C)"
