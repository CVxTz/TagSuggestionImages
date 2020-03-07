import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from imageio import imread

st.title("Sample UI")

file = st.file_uploader("Upload file", type=['jpg'])

if file:
    img = imread(file)

    st.write(img)

    data = pd.DataFrame(
        {"x": ['R', 'G', 'B'], 'y': [np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])]})

    bars = alt.Chart(data).mark_bar().encode(
        x='y',
        y="x"
    )

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='y'
    )

    (bars + text).properties(height=900)

    st.write(bars)
