FROM python:3.6-slim
COPY app/main.py /deploy/
COPY app/config.yaml /deploy/
WORKDIR /deploy/
RUN wget -O repo.zip http://github.com/CVxTz/TagSuggestionImages/zipball/master/
RUN pip install repo.zip
EXPOSE 8080

ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1