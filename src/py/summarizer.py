import ollama
import json
from typing import Dict, List
from .utils import *


def text_summarize(text: str, content_type: str) -> str:
    """
    Summarizes the provided text based on the specified content type.

    Parameters:
    text (str): The text to be summarized.
    content_type (str): The type of the content which must be 'job', 'course', or 'scholarship'.

    Returns:
    str: The summarized text.

    Raises:
    ValueError: If the content_type is invalid.
    TypeError: If any input is not of the expected type.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    if not isinstance(content_type, str):
        raise TypeError("Content type must be a string.")
    valid_types = ['job', 'course', 'scholarship']
    if content_type not in valid_types:
        raise ValueError(f"content_type must be one of {valid_types}, got '{content_type}' instead.")
    # Compute the number of words in the original text
    text_length = count_words(text)
    # Calculate the summary length based on the text length
    sum_length = calculate_summary_length(text_length)
    # Generate the appropriate summarization prompt
    prompts = hydrate_summary_prompt(text, sum_length, content_type)
    # Get the summary from an external summarization function, passing the prompt
    summary = call_summarize(prompts)
    return summary
