FROM ollama/ollama

RUN apt-get update && apt-get install -y git curl

RUN curl localhost:11434
RUN ollama list
RUN ollama pull llama3

