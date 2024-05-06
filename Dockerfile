# Base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code 

# Copy the requirements.txt file to the working directory
COPY ./requirements.txt ./

# Install git and curl
RUN apt-get update && apt-get install git -y && apt-get install curl -y

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the working directory
COPY ./src ./src

COPY ./run_ollama.sh ./

RUN chmod +x run_ollama.sh \
    && ./run_ollama.sh

# Start the application using uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
