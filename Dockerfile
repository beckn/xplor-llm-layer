FROM ollama/ollama

EXPOSE 11434
RUN ollama serve & 

RUN sleep 15
RUN ollama list
RUN ollama pull llama3

