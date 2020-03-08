import sys
import matplotlib.pyplot as plt
import altair as alt
import os

sys.path.append("image_tag_suggestion")

import streamlit as st
import predictor

st.title("Image Tag Suggestion")

file = st.file_uploader("Upload file", type=["jpg"])

predictor_config_path = (
    "config.yaml"
    if os.path.isfile("config.yaml")
    else "image_tag_suggestion/config.yaml"
)

image_predictor = predictor.ImagePredictor.init_from_config_url(predictor_config_path)
label_predictor = predictor.LabelPredictor.init_from_config_url(predictor_config_path)

if file:
    pred, arr = image_predictor.predict_from_file(file)
    plt.imshow(arr)
    plt.axis("off")
    st.pyplot()
    data = label_predictor.predict_dataframe_from_array(pred)

    # st.write(arr)

    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(x="scores:Q", y=alt.X("label:O", sort=data["label"].tolist()),)
    )

    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text="label")

    (bars + text).properties(height=900)

    st.write(bars)
