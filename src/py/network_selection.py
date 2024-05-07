import ollama
import json
from typing import Dict, List
from .utils import *


def network_selection_service(search_item :str) -> dict:
    """
     Identifies the network based on the given search item.

     Parameters:
     search_itemn (str): The item for which the network needs to be identified.
     

     Returns:
     list[dict]: A dictionaries containing the identified language for the given state and country.
     """

    if not isinstance(search_item, str):
        raise TypeError("search_item must be a string.")
   
    network_identified_prompt = hydrate_network_prompt( search_item )
    network_identified = call_network_identification(network_identified_prompt)

    return network_identified


def clear_cache_network_identification():
    network_identification.cache_clear()
    return True