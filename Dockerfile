FROM ollama/ollama

COPY ./run-ollama.sh /tmp/run-ollama.sh

WORKDIR /tmp

RUN chmod +x run-ollama.sh \
    && ./run-ollama.sh

EXPOSE 11434

#EXPOSE 11434
#RUN ollama serve & 

#RUN ollama pull llama3

