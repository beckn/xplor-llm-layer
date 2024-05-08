from fastapi import FastAPI, HTTPException
import json
from datetime import datetime
from pydantic import BaseModel, validator
from typing import Dict, List, Union
import subprocess

from .py.language_selection import language_selection_service, clear_cache_language_identification
from .py.summarizer import text_summarize, clear_cache_text_summarize
from .py.reviews_analyser import review_analyser, clear_cache_review_analyser
from .py.network_selection import network_selection_service, clear_cache_network_identification
import time

app = FastAPI(
    title="Xplor - LLAMA3 Services API",
    description="This API provides access to the LLAMA3 services for summarization, review analysis, and language "
                "selection based on location.",
    version="v0.0.1",
    contact={
        "name": "WITSLAB"
    },
    docs_url="/docs",
    redoc_url="/redocs",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=[{"name": "Health Check", "description": "Healthcheck operations"},
                  {"name": "Summarise", "description": "Summarisation operations"},
                  {"name": "Review Analyser", "description": "Review Analyse operations"},
                  {"name": "Location Based language Selection", "description": "Language Selection operations"},
                  {"name": "Search Based Network Selection", "description": "Network Selection operations"}
                  ]

)



current_datetime = datetime.now()


#################################################################################################################
#                                   Health Check                                                                #
#################################################################################################################


@app.get("/healthcheck", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify the status of the application.
    """

    return {"status": "ok"}


@app.get("/datecheck", tags=["Health Check"])
def date_check():
    """
    Health check endpoint to verify the status of the application.
    """

    return {"date": current_datetime}




#################################################################################################################
#                                   Clear Cache                                                                 #
#################################################################################################################

@app.post("/clear_cache/", tags=["Health Check"])
async def clear_cache():
    summary = clear_cache_text_summarize()
    review = clear_cache_review_analyser()
    language = clear_cache_language_identification()
    network = clear_cache_network_identification()
    if summary:
        a = "Summary Cache Cleared"
    else:
        a = "Summary Cache Not Cleared"
    if review:
        b = "Review Cache Cleared"
    else:
        b = "Review Cache Not Cleared"
    if language:
        c = "Language Cache Cleared"
    else:
        c = "Language Cache Not Cleared"
    if network:
        d = "Network Cache Cleared"
    else:
        d = "Network Cache Not Cleared"
    return {"status": a + b + c + d }



#################################################################################################################
#                                   Summarize                                                                   #
#################################################################################################################


class SummaryRequest(BaseModel):
    text: str
    content_type: str

    # Validator to ensure content_type is one of the accepted values
    @validator('content_type')
    def validate_content_type(cls, v):
        valid_types = ['job', 'course', 'scholarship']
        if v not in valid_types:
            raise ValueError(f"content_type must be one of {valid_types}")
        return v


@app.post('/summarise/', tags=["Summarise"])
async def create_summary(request: SummaryRequest):
    try:
        # Use the utility function to summarize the text
        summary = text_summarize(request.text, request.content_type)
        return {"summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch any other exceptions and return a generic error message
        raise HTTPException(
            status_code=500, detail="An error occurred during summarization.")

'''
import requests

# Define the URL of the API endpoint
url = 'http://localhost:8000/summarize/'

# Define the headers
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Define the payload as a dictionary, which will be automatically converted to JSON by requests
payload = {
    "text": "Your example text here...",
    "content_type": "job"
}

# Make the POST request
response = requests.post(url, json=payload, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Print the response body (the summarized text)
    print(response.json())
else:
    # Print the error
    print("Failed to summarize:", response.status_code, response.text)

'''


#################################################################################################################
#                                   Review Analysis                                                             #
#################################################################################################################


class ReviewAnalyser(BaseModel):
    reviews: Union[str, List[str]]


@app.post('/review_analysis', tags=["Review Analyser"])
async def create_review_analyser(input_data: ReviewAnalyser):
    try:
        # Convert input data to a single string
        if isinstance(input_data.reviews, list):
            # Join list elements into a single string separated by spaces
            combined_reviews = " ".join(input_data.reviews)
        else:
            combined_reviews = input_data.reviews

        # Pass the combined string to the review_analyser function
        summary = review_analyser(combined_reviews)
        return {"summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch any other exceptions and return a generic error message
        raise HTTPException(status_code=500, detail=f"An error occurred during review analysis: {str(e)}")


#################################################################################################################
#                                   Location Based Language                                                     #
#################################################################################################################

class LocationRequest(BaseModel):
    city : str
    state: str
    country: str


@app.post('/language_selection', tags=['Location Based language Selection'])
async def language_selection(request: LocationRequest):
    try:
        # Use the utility function to summarize the text
        language = language_selection_service(request.city, request.state, request.country)
        return {"languages": language}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch any other exceptions and return a generic error message
        raise HTTPException(
            status_code=500, detail="An error occurred during fetching language.")
    
#################################################################################################################
#                                   Location Based Language                                                     #
#################################################################################################################

class NetworkRequest(BaseModel):
    search_item : str


@app.post('/network_selection', tags=['Search Based Network Selection'])
async def network_selection(request: NetworkRequest):
    try:
        # Use the utility function to summarize the text
        network = network_selection_service(request.search_item)
        return network
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch any other exceptions and return a generic error message
        raise HTTPException(
            status_code=500, detail="An error occurred during fetching network.")