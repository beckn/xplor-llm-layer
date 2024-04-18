import ollama
import json
from .models import SummaryRequest




messages = []


def summarize(input_data: SummaryRequest):
    """
    Summarizes the input text.

    Args:
        input_data (SummaryRequest): The input text to be summarized.

    Returns:
        dict: A JSON object containing the summary (length is 10% of input text).
    """
    
    word_count = len(input_data.text.split())
    summary_length = int(word_count * 0.1)  # 10% of the word count

    messages.clear()
    stream = ollama.chat(
        model='llama2',
        messages=[{
            'role': 'user',
            'content': f"Summarise the following text in atleast {summary_length} words: {input_data.text}",
        }],
        stream=True,
    )
    for chunk in stream:
        input_data = chunk['message']['content']
        messages.append(input_data)
    full_message = ''.join(messages)
    return json.dumps({'Summary': full_message})
