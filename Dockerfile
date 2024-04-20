# Use an official Python runtime as a parent image
FROM python:3.8
 
#Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama2 

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
