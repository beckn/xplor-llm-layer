import ollama
from fastapi import FastAPI, HTTPException
import json
from datetime import datetime
from pydantic import BaseModel, validator
from typing import Dict, List, Union
import subprocess
from .py.summarizer import text_summarize
from .py.reviews_analyzer import reviewsanalysis
import time


app = FastAPI(
    title="Xplor - LLAMA2 Summarizer API",
    description="Summarize your text using LLAMA2",
    version="v0.0.1",
    contact={
        "name": "WITSLAB"
    },
    docs_url="/docs",
    redoc_url="/redocs",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=[{"name": "healthcheck", "description": "Healthcheck operations"},
                  {"name": "summarize", "description": "Summarization operations"},
                  {"name": "review_analyser",
                      "description": "Review Analyse operations"}
                  ]

)

# Pull the llama2 model from the server
command_to_serve = "ollama serve"
command_to_pull = "ollama pull llama2"

# Start the 'ollama serve' command in the background
serve_process = subprocess.Popen(command_to_serve, shell=True)

# Wait for a short period to ensure 'ollama serve' has started
time.sleep(5)

# Execute 'ollama pull llama2'
pull_process = subprocess.Popen(command_to_pull, shell=True)

# Wait for 'ollama pull llama2' to complete
pull_process.wait()


current_datetime = datetime.now()
#################################################################################################################
#                                   Health Check                                                                #
#################################################################################################################


@app.get("/healthcheck", tags=["healthcheck"])
def health_check():
    """
    Health check endpoint to verify the status of the application.
    """

    return {"status": "ok"}


@app.get("/datecheck", tags=["healthcheck"])
def date_check():
    """
    Health check endpoint to verify the status of the application.
    """

    return {"date": current_datetime}

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


@app.post('/summarize/', tags=["summarize"])
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


@app.post('/reviewsanalysis')
def reviewsanalysis_endpoint(input_data: SummaryRequest):
    return reviewsanalysis(input_data)
