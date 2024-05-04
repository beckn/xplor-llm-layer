## Utils file 
## Contains all the reusable functions 

import ollama
import json
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
logging.basicConfig(filename='../app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
#                                   UTILS FOR Summary                                                       #
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
            summary_length = max(int(0.3 * text_length),
                                 int(0.5 - (text_length - 300) / 3500 * text_length))  # Scale down to 30%.
        else:
            summary_length = max(int(0.1 * text_length),
                                 int(0.3 - (text_length - 1000) / 9000 * text_length))  # Scale down to 10%.
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
    # Specific instructions based on content type
    content_specific_prompt = {
        'job': "Your task is to summarize this job description. Focus on key responsibilities, required qualifications, and employment benefits.",
        'course': "Your task is to summarize this online course description. Highlight the main learning objectives, course outline, and target audience.",
        'scholarship': "Your task is to summarize the scholarship details. Include important eligibility criteria, scholarship benefits, and application deadlines."
    }
    user_prompt = {
        'role': 'user',
        'content': f"""You are an expert summarizer capable of understanding the content and summarizing aptly, keeping most valid information intact.
                   Develop a summarizer that efficiently condenses the text into a concise summary. 
                   The summaries should capture essential information and convey the main points clearly and accurately. 
                   The summarizer must be able to handle content related to {content_type}s. 
                   It should prioritize key facts, arguments, and conclusions while maintaining the integrity and tone of the original text. 
                   Aim for a summary that is approximately {sum_length} words of the size. 
                   Focus on clarity, brevity, and relevance to ensure the summary is both informative and readable. 
                   The text is as follows: {text} {content_specific_prompt[content_type]}
                   
                   Provide just the summary nothing else. No preceeding sentences or succeeding sentences.
                   Dont leave any notes at the end that this is a summary.
                   """
    }
    return [user_prompt]


# Example usage:
#content_text = "This course offers an in-depth exploration of modern data sciences, covering key concepts, applications, and tools. It is ideal for professionals seeking to enhance their understanding of data analysis and machine learning."
#prompt = hydrate_summary_prompt(content_text, 30, 'course')
#print(prompt)
def hydrate_language_prompt(state: str, country: str):
    user_prompt = {
        'role': 'user',
        'content': f""" List of language spoken in {state} in {country} (return json with just two keys - language, 
        percentage with % sign) in descending order.No preceding sentences or succeeding 
        sentences. Dont leave any notes at the end."""

    }
    return [user_prompt]


def language_identification(prompts: List[Dict[str, str]]) -> dict:
    try:
        response = ollama.chat(
            model='llama3',
            messages=prompts,
            stream=False,
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        raise RuntimeError("Failed to identify language due to an external API error: " + str(e))


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
            model='llama3',
            messages=prompts,
            stream=False,
        )
        # Assuming the summary is correctly formatted in the response under 'message' and 'content'
        return response['message']['content']
    except Exception as e:
        raise RuntimeError("Failed to generate summary due to an external API error: " + str(e))


#summarize("""  To make it easy for you to get started with GitLab, here's a list of recommended next steps.  """)


#############################################################################################################
#############################################################################################################
#                                   UTILS FOR Reviews Analysis                                              #
#############################################################################################################
#############################################################################################################


def process_text(input_text: str):
    """
    Processes input text, which can be either a JSON string or plain text. For JSON,
    it ensures the string is no more than 25,000 characters, trimming without breaking the JSON structure.
    For plain text, it cuts the text to 25,000 characters or less without breaking words if possible.

    Args:
    input_text (str): The text to process, which can be in JSON format or plain text.

    Returns:
    str: Processed text with 25,000 characters or fewer.
    """
    try:
        # Attempt to load the text as JSON
        data = json.loads(input_text)
        is_json = True
    except json.JSONDecodeError:
        # Handle plain text
        is_json = False

    if is_json:
        # It's JSON, handle as JSON
        json_string = json.dumps(data)
        total_chars = len(json_string)
        if total_chars > 25000:
            keep_chars = min(25000, total_chars // 3)
            trimmed_json_string = json_string[:keep_chars].rsplit('}', 1)[0] + '}'
            try:
                valid_json = json.loads(trimmed_json_string)
                return json.dumps(valid_json)
            except json.JSONDecodeError:
                print("Error in trimming JSON. Adjusting trimming logic may be necessary.")
                return "{}"
    else:
        # Handle as plain text
        total_chars = len(input_text)
        if total_chars > 25000:
            keep_chars = min(25000, total_chars // 3)
            # Try to avoid breaking words
            if ' ' in input_text[keep_chars:] and keep_chars < total_chars:
                trimmed_text = input_text[:keep_chars].rsplit(' ', 1)[0]
            else:
                trimmed_text = input_text[:keep_chars]
            return trimmed_text
        else:
            return input_text


def hydrate_review_analyser_prompt(text: str):
    user_prompt = {
        'role': 'user',
        'content': f"""You are an advanced language model specifically trained for deep text analysis and synthesis. 
                Your primary function today is to assess a range of customer reviews for a given product, extracting pivotal sentiments, pinpointing common concerns, and identifying frequent praises. 
                Your ultimate goal is to condense these findings into a singular, well-crafted summary that encapsulates the overall consumer experience with the product. 
                This summary should remain unbiased, objective, and focus strictly on the product features, overall quality, and user satisfaction.
                It should always contain the overall sentiment. Either positive, negative or neutral. 
                Make the summary only 30 word long

                Guidelines for Summary Creation:
                1. **Identify Common Themes**: Concentrate on the aspects of the product that are consistently mentioned across multiple reviews. This includes highlighting the most praised features as well as the most critiqued ones.
                2. **Objective Language**: Maintain an objective tone throughout the summary. Avoid inserting personal opinions or artistic flourishes that could skew the information or imply subjectivity.
                3. **Professionalism in Communication**: Use a professional and respectful language style. The summary should read as if it were part of an official product description or a customer service response.
                4. **Inclusivity and Respect**: Steer clear of any language that could be considered profane, colloquial, or sensitive to cultural contexts. The summary must be suitable for a diverse audience.
                5. **Actionable Insights**: Craft the summary in a way that provides clear, actionable insights to potential buyers. They should be able to understand the most significant pros and cons of the product at a glance, aiding them in their decision-making process.

                These are the reviews which needs to be summarised. 
                {text}
                Generate just a Summary. Dont include anything else. 
                   """
    }
    return [user_prompt]


def analyse(prompts: List[Dict[str, str]]) -> str:
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
            model='llama3',
            messages=prompts,
            stream=False,
        )
        # Assuming the summary is correctly formatted in the response under 'message' and 'content'
        return response['message']['content']
    except Exception as e:
        raise RuntimeError("Failed to generate review due to an external API error: " + str(e))
