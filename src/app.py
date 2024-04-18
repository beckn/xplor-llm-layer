import ollama
import fastapi
import json
from pydantic import BaseModel
import subprocess
from .py.summarizer import summarize
from .py.reviews_analyzer import reviewsanalysis
from .py.models import SummaryRequest

app = fastapi.FastAPI(
    title="Xplor - LLAMA2 Summarizer API",
    description="Summarize your text using LLAMA2",
    version="v0.0.1",
    contact={
        "name": "WITSLAB"
    },
    docs_url="/docs",
    redoc_url="/redocs",
)

# Pull the llama2 model from the server
command = "ollama pull llama2"

# Execute the command
process = subprocess.Popen(command, shell=True)




@app.post('/summarize')
def summarize_endpoint(input_data: SummaryRequest):
   return summarize(input_data)


@app.post('/reviewsanalysis')
def reviewsanalysis_endpoint(input_data: SummaryRequest):
    return reviewsanalysis(input_data)
