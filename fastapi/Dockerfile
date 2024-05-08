FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code 

# Copy the requirements.txt file to the working directory
COPY ./requirements.txt ./

# Install git and curl
RUN apt-get update && apt-get install -y git curl

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN touch app.log

# Copy the source code to the working directory
COPY ./src ./src

# Start the application using uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]