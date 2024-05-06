import ollama
import json
from typing import Dict, List
from .utils import *

def review_analyser(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    # Cut the length of input data to predefined size
    cut_text = process_text(text) 
    # Generate the appropriate summarization prompt
    prompts = hydrate_review_analyser_prompt(cut_text)
    # Get the summary from an external summarization function, passing the prompt
    summary = call_analyse(prompts)
    return summary

