# Xplor - LLAMA2 Summarizer API

This is a FastAPI application that uses the LLAMA2 model to summarize text.

## Features

- Summarize text.
- Analyze the sentiment of reviews and categorize them as positive or negative.

## Setup

1. Clone the repository.
2. Install Ollama for your required OS:
### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows preview

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```
3. Install the required Python packages:
   
```sh
pip install -r requirements.txt
```


4. Run the application:

```sh
uvicorn src.app:app --reload
```

## Usage

The API has the following endpoints:

- `POST /summarize`: Summarizes the input text.
- `POST /reviewsanalysis`: Analyzes the sentiment of the input reviews.

Each endpoint accepts a JSON object with the following structure:

```json
{
    "text": "your message here"
}
```

## Authors

[WITSLAB](https://www.thewitslab.com/)


## License

Pending Discussion
