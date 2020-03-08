FROM python:3.6-slim
COPY image_tag_suggestion/main.py image_tag_suggestion/preprocessing_utilities.py /deploy/
COPY image_tag_suggestion/predictor.py image_tag_suggestion/utils.py /deploy/
COPY image_tag_suggestion/config.yaml /deploy/
COPY image_tag_suggestion/image_representation.h5 /deploy/
# Download from https://github.com/CVxTz/TagSuggestionImages/releases
COPY image_tag_suggestion/labels.json /deploy/
# Download from https://github.com/CVxTz/TagSuggestionImages/releases
COPY requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8501

ENTRYPOINT streamlit run main.py