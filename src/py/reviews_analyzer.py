# src/py/reviews_analyzer.py

import ollama
import json
from .models import SummaryRequest

reviews = []

def reviewsanalysis(input_data: SummaryRequest):
    """
    Analyzes the sentiment of the input reviews and categorizes them as positive or negative.

    Args:
        input_data (SummaryRequest): The input reviews to be analyzed.

    Returns:
        dict: A JSON object containing the analysis result.
    """
    
    reviews.clear()
    stream = ollama.chat(
        model='llama2',
        messages=[{
            'role': 'user',
            'content': f"Categorise the following reviews in just one word positive or negative : {input_data.text}",
        }],
        stream=True,
    )
    for chunk in stream:
        input_data = chunk['message']['content']
        reviews.append(input_data)
    reviewsfinal = ''.join(reviews)
    return json.dumps({'Reviews Analysis': reviewsfinal})


