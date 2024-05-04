# Xplor - LLAMA3 Services

This is a FastAPI application that uses the LLAMA3 model to summarize text, analyze reviews, and select languages based on location.

## Features

- Summarize text.
- Analyze the sentiment of reviews and categorize them as positive or negative.
- Select language based on location.

## Setup

1. Clone the repository.
2. Install Ollama for your required OS:
### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows preview

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux
```sh 
curl -fsSL https://ollama.com/install.sh | sh
```
3. Install the required Python packages:

```sh
pip install -r requirements.txt
```
### Run the application:

```sh
uvicorn src.app:app --reload
```
## Run the application using docker
```sh
docker build -t llama3-services .
docker run -p 8000:8000 llama3-services
```

## Usage

The API has the following endpoints:

- `POST /summarize`: Summarizes the input text.
- `POST /reviewsanalysis`: Analyzes the sentiment of the input reviews.
- `POST /language_selection`: Selects the language based on the input location.
Each endpoint accepts a JSON object with the following structure:
```sh
{
    "text": "your message here",
    "content_type": "["course", "scholarship", "jobs"]"
}
```
content_type accepts a string from the above list.

 For the language selection endpoint, the JSON object structure is:

```sh
{
    "state": "your state here",
    "country": "your country here"
}
```
## Authors

[WITSLAB](https://www.thewitslab.com/)


## License

Pending Discussion
