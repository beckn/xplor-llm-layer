from .utils import *
import json
from typing import Dict, List

def language_selection_service(city:str, state: str, country: str) -> dict:
    """
     Identifies the language based on the given state and country.

     Parameters:
     state (str): The state for which the language needs to be identified.
     country (str): The country for which the language needs to be identified.

     Returns:
     list[dict]: A list of dictionaries containing the identified language for the given state and country.
     """

    if not isinstance(city, str):
        raise TypeError("City must be a string.")
    if not isinstance(state, str):
        raise TypeError("State must be a string.")
    if not isinstance(country, str):
        raise TypeError("Country must be a string.")

    language_identified_prompt = hydrate_language_prompt(city, state, country)
    language_identified = call_language_identification(language_identified_prompt)

    return language_identified


def clear_cache_language_identification():
    language_identification.cache_clear()
    return True