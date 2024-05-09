FROM ollama/ollama

EXPOSE 11434
RUN /bin/ollama serve & 

RUN sleep 15
RUN /bin/ollama list
RUN /bin/ollama pull llama3

