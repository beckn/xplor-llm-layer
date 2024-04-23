## Utils file 
## Contains all the reusable functions 

import ollama
import logging
import time
from datetime import datetime
import pandas as pd
from functools import lru_cache, wraps
import warnings
from typing import Dict, List
warnings.filterwarnings("ignore")


#############################################################################################################
#############################################################################################################
#                                   UTILS FOR General Purpose                                               #
#############################################################################################################
#############################################################################################################



# Configure logging
logging.basicConfig(filename='../app.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_function_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log the demarcation 
        logging.info(f"--------------------------------------------------------------------")
        # Log the function's input (arguments and keyword arguments) & function's output
        logging.info(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs} and function returned:  {result} ")    
        
        # Log the execution time
        logging.info(f"{func.__name__} execution time: {execution_time:.4f} seconds")

        return result
    return wrapper

# Example usage of the decorator
#@log_function_data
#def sample_function(a, b):
#    """Example function that adds two numbers."""
#    return a + b

# Call the decorated function
#result = sample_function(5, 7)


#############################################################################################################
#############################################################################################################
#                                   UTILS FOR Reviews Analysis                                              #
#############################################################################################################
#############################################################################################################


def count_words(text):
    """
    Counts the number of words in a given text string.
    
    Parameters:
    text (str): The text string from which to count words.
    
    Returns:
    int: The number of words in the text.
    
    Raises:
    TypeError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    # Normalize the text to ensure consistent splitting
    # This removes extra spaces and handles different types of whitespace
    normalized_text = ' '.join(text.split())
    # Split the normalized text into words based on whitespace
    words = normalized_text.split()
    # Return the number of words
    return len(words)


#############################################################################################################
def calculate_summary_length(text_length):
    """
    Determine the appropriate summary length based on the length of the input text.
    
    Parameters:
    text_length (int): The length of the input text in terms of number of words.
    
    Returns:
    int: The recommended number of words for the summary.
    """
    if not isinstance(text_length, int) or text_length < 0:
        raise ValueError("Input must be a non-negative integer representing the word count of the text.")
    ## 
    if text_length == 0:
        return 0  # No words to summarize if the text length is 0.
    unchanged_threshold = 50
    summary_length = text_length  # Default to full length if no rules apply.
    if text_length > unchanged_threshold:
        if text_length < 300:
            summary_length = int(0.5 * text_length)  # 50% of original for short texts.
        elif text_length < 1000:
            summary_length = max(int(0.3 * text_length), int(0.5 - (text_length - 300) / 3500 * text_length))  # Scale down to 30%.
        else:
            summary_length = max(int(0.1 * text_length), int(0.3 - (text_length - 1000) / 9000 * text_length))  # Scale down to 10%.
    return summary_length



#############################################################################################################


'''

in any other py file apart from app.py. Make it so that its just a text input. We use data classes for api classes more. 

For now just keep str input in two .py files and use a class in app.py


'''

    
def hydrate_summary_prompt(text: str, sum_length: int, content_type: str):
    """
    Generates a customized summarization prompt based on the type of content.

    Parameters:
    text (str): The text to be summarized.
    sum_length (int): Target length of the summary in words.
    content_type (str): Type of the content which must be 'job', 'course', or 'scholarship'.

    Returns:
    dict: A dictionary with system and user prompts tailored for the specific content type.
    """
    if content_type not in ['job', 'course', 'scholarship']:
        raise ValueError("content_type must be one of 'job', 'course', or 'scholarship'")
    # General part of the prompt that remains constant
    system_prompt = {
        'role': 'system',
        'content': "You are an expert summarizer capable of understanding the content and summarizing aptly, keeping most valid information intact."
    }
    # Specific instructions based on content type
    content_specific_prompt = {
        'job': "Your task is to summarize this job description. Focus on key responsibilities, required qualifications, and employment benefits.",
        'course': "Your task is to summarize this online course description. Highlight the main learning objectives, course outline, and target audience.",
        'scholarship': "Your task is to summarize the scholarship details. Include important eligibility criteria, scholarship benefits, and application deadlines."
    }
    user_prompt = {
        'role': 'user',
        'content': f"""Develop a summarizer that efficiently condenses the text into a concise summary. 
                   The summaries should capture essential information and convey the main points clearly and accurately. 
                   The summarizer must be able to handle content related to {content_type}s. 
                   It should prioritize key facts, arguments, and conclusions while maintaining the integrity and tone of the original text. 
                   Aim for a summary that is approximately {sum_length}% of the original size. 
                   Focus on clarity, brevity, and relevance to ensure the summary is both informative and readable. 
                   The text is as follows: {text} {content_specific_prompt[content_type]}
                   
                   Provide just the summary nothing else. No preceeding sentences or succeeding sentences.
                   Dont leave any notes at the end that this is a summary.
                   """
    }
    return [system_prompt, user_prompt]

# Example usage:
#content_text = "This course offers an in-depth exploration of modern data sciences, covering key concepts, applications, and tools. It is ideal for professionals seeking to enhance their understanding of data analysis and machine learning."
#prompt = hydrate_summary_prompt(content_text, 30, 'course')
#print(prompt)



def summarize(prompts: List[Dict[str, str]]) -> str:
    """
    Calls an external API or model to generate a summary based on the given prompts.

    Parameters:
    prompts (List[Dict[str, str]]): A list of dictionaries that contain the prompts for summarization.

    Returns:
    str: The summarized text.
    """
    try:
        # Assuming the ollama chat function accepts a list of prompts formatted as required
        response = ollama.chat(
            model='llama2',
            messages=prompts,
            stream=False,
        )
        # Assuming the summary is correctly formatted in the response under 'message' and 'content'
        return response['message']['content']
    except Exception as e:
        raise RuntimeError("Failed to generate summary due to an external API error: " + str(e))


#summarize("""  To make it easy for you to get started with GitLab, here's a list of recommended next steps.  """)