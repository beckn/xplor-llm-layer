
import json
from typing import Dict, List

from src.py.utils import *


@log_function_data
# Path: src/py/categories_selection.py
def generate_domain_categories_service(domain: str) -> dict:

    if not isinstance(domain, str):
        raise TypeError("Domain must be a string.")

    categories_identified_prompt = hydrate_categories_prompt(domain)
    categories_identified = categories_identification(categories_identified_prompt)

    return categories_identified
