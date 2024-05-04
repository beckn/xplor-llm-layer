from .utils import *


def language_selection_service(state: str, country: str) -> dict:
    """
     Identifies the language based on the given state and country.

     Parameters:
     state (str): The state for which the language needs to be identified.
     country (str): The country for which the language needs to be identified.

     Returns:
     list[dict]: A list of dictionaries containing the identified language for the given state and country.
     """

    if not isinstance(state, str):
        raise TypeError("State must be a string.")
    if not isinstance(country, str):
        raise TypeError("Country must be a string.")

    language_identified_prompt = hydrate_language_prompt(state, country)
    language_identified = language_identification(language_identified_prompt)

    return language_identified
