FROM ollama/ollama
EXPOSE 11434
RUN ollama serve & 
RUN sleep 40
RUN ollama list
RUN ollama pull llama3
CMD ["serve"]