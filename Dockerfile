FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code 

# Copy the requirements.txt file to the working directory
COPY ./requirements.txt ./

# Install git and curl
RUN apt-get update && apt-get install -y git curl

RUN df -h
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for the model
RUN mkdir llama3
RUN touch app.log

# Login to Hugging Face and download the model to the specified directory
RUN huggingface-cli login --token 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 &&  huggingface-cli download meta-llama/Meta-Llama-3-8B --include "*.safetensors" --include "*.json" --local-dir /code/llama3

RUN ls -lah /code/llama3
RUN cd /code/llama3
RUN pwd
# Copy the source code to the working directory
COPY ./src ./src

# Start the application using uvicorn
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port 8000 >> app.log 2>&1"]