Code for: https://medium.com/analytics-vidhya/descriptive-image-tag-suggestion-in-a-streamlit-app-a17f09b49c4e

## Docker
```
sudo docker build -t img_tag_suggesting .
```

```
docker run -p 8501:8501 img_tag_suggesting
```

### Descriptive Image Tag Suggestion In a Streamlit App

Build a Web-based Image tag suggestion app using Tensorflow and Streamlit

We will build a system that can do automatic tag suggestions for images using a
vision model. It means that having an image as input, it will predict ranked
list of labels that describe this image. This can be useful for image search or
recommendation applied to an image collection.<br> This project will be based on
an amazing image dataset called Open Images V6:
[https://storage.googleapis.com/openimages/web/download.html](https://storage.googleapis.com/openimages/web/download.html).
It has 7,337,077 images with bounding boxes, class information, and image-level
labels.

Each image out of the 7,337,077 million has one or multiple labels associated
with it from a set with a total 19,958 labels

![](https://cdn-images-1.medium.com/max/600/1*qZBnDsvSbioBdpEYqpy5TQ.jpeg)
<span class="figcaption_hack">Image from Unsplash</span>

For example, this image would have labels like trees, snow, sky … These types of
labels can be used as weak supervision to do build a vision model that tries to
predict the tags that best describe an image.

<br>

<br>

<br>

<br>

### Model

The model used here is very similar to the one I described in one of the earlier
posts (
[https://towardsdatascience.com/building-a-deep-image-search-engine-using-tf-keras-6760beedbad](https://towardsdatascience.com/building-a-deep-image-search-engine-using-tf-keras-6760beedbad)
).

The model used has one MobileNetV2 sub-model that encodes each image into a (50,
1) vector and then an embedding sub-model that encodes a positive label and a
negative label into two separate (50, 1) vectors.

We use the Triplet Loss where the objective is to pull the image representation
and the embedding of the positive label closer together.

![](https://cdn-images-1.medium.com/max/800/1*5gnodeMtIGm0TaP5Y1Itug.png)
<span class="figcaption_hack">Puppy image modified from Unsplash</span>

The image sub-model produces a representation for the Anchor **E_a **and the
embedding sub-model outputs the embedding for the positive label **E_p** and the
embedding for the negative label **E_n**.

We then train by optimizing the following triplet loss:

**L = max( d(E_a, E_p)-d(E_a, E_n)+alpha, 0)**

Where d is the euclidean distance and alpha is a hyperparameter equal to 0.4 in
this experiment.

Basically what this loss allows to do is to make **d(E_a, E_p) **small and
make** d(E_a, E_n) **large, so that each image representation is close to the
embedding of its label and far from the embedding of a random label.

When doing the prediction we compute the representation of the image once and
compute its distance to each label embedding. We then convert the distances to
“scores” and sort the scores from highest to lowest. We return the top k highest
scoring labels.

### Building the UI

We will use Streamlit python library to build a web application that allows us
to upload a jpg image and then receive the top 20 most likely labels.

Streamlit makes it easy to build a “demo”-like application built in python
directly from the browser.

The use of this package is very easy. What we want to do is :

* Upload an image file.
* Predict the top 20 most likely labels for the image.
* Display the results in a nice plot.

First, we load our predictor classes :

    image_predictor = predictor.ImagePredictor.init_from_config_url(predictor_config_path) 
    label_predictor = predictor.LabelPredictor.init_from_config_url(predictor_config_path)

1.  Upload an image file :
```

    import streamlit as st
    import matplotlib.pyplot as plt # To plot the image
    import altair as alt # To plot the label ranking

    file = st.file_uploader("Upload file", type=["jpg"])
```

2. Predict Top 20 labels :
```

    if file:
        # Compute image representation
        pred, arr = image_predictor.predict_from_file(file)
        plt.imshow(arr)
        plt.axis("off")
        # Plot the image to the web page
        st.pyplot()
        # predict the labels
        data = label_predictor.predict_dataframe_from_array(pred)
```

3. Display the results :
```
    bars = (
            alt.Chart(data)
            .mark_bar()
            .encode(x="scores:Q", y=alt.X("label:O", sort=data["label"].tolist()),)
        )

    text = bars.mark_text(
            align="left",
            baseline="middle",
            dx=3,
        ).encode(text="label")

    (bars + text).properties(height=900)

    st.write(bars)
```

Done!

The result :

![](https://cdn-images-1.medium.com/max/800/1*h5PhlvuwuEtRO8if3KxUmw.png)
<span class="figcaption_hack">Trees Image from Unsplash</span>

![](https://cdn-images-1.medium.com/max/800/1*-JrVxd9Jvg5PGD-QpuUqTg.png)
<span class="figcaption_hack">Prediction plot</span>

Some of the suggestions are spot-on like Tree, Plant or Land plant but others
are only so-so, I guess handling 19,000 possible labels is too much for a tiny
MobileNet 😅.

#### Run with Docker

You can easily run this app locally using docker. Just clone the repo referenced
at the end of the post and build this docker image :

    FROM python:3.6-slim
    COPY image_tag_suggestion/main.py image_tag_suggestion/preprocessing_utilities.py /deploy/
    COPY image_tag_suggestion/predictor.py image_tag_suggestion/utils.py /deploy/
    COPY image_tag_suggestion/config.yaml /deploy/
    COPY image_tag_suggestion/image_representation.h5 /deploy/
    # Download from 
    COPY image_tag_suggestion/labels.json /deploy/
    # Download from 
    COPY requirements.txt /deploy/
    WORKDIR /deploy/
    RUN pip install -r requirements.txt
    EXPOSE 8501

    ENTRYPOINT streamlit run main.py

Then build and run :


#### Deploy on Heroku

Heroku allows you to deploy a python app directly from your GitHub repo.<br> You
just need to specify Three files :

* setup.sh: Helper file that downloads the models and sets some parameters for
streamlit.
* runtime.txt: Specifies the python version you want to use.
* Procfile: Specifies the type of application and command to run it.

All of those files are available in the Github Repo linked at the end of this
page.

Then you just need to create a free account on Heroku and follow those steps :

* Create the app :

![](https://cdn-images-1.medium.com/max/800/1*JC2eU0KmXyOF_iMdmUmR8A.png)
<span class="figcaption_hack">Create App</span>

* Pick App name :

![](https://cdn-images-1.medium.com/max/800/1*BtvMAeomo0UWaDVVj7Q9tQ.png)
<span class="figcaption_hack">Name</span>

* Specify Github repo :

![](https://cdn-images-1.medium.com/max/800/1*-jk2IMiAF_jbZMEO63tn0w.png)

* Choose a branch and deploy :

![](https://cdn-images-1.medium.com/max/800/1*Op_X7sxnN1wO3skL7JfUiw.png)
<span class="figcaption_hack">Deploy</span>

* Tadaaaa!

![](https://cdn-images-1.medium.com/max/1200/1*KnT-YP5Bh38iu8B9MdJcDg.png)
<span class="figcaption_hack">My cat</span>

At least it got Kitten, Cat Toy, and Carnivore right in the top 20 tags 😛.

### Conclusion

In this project, we built an application with a web UI and can predict the top
descriptive tags that best fit an image. The machine learning part still needs
some improvements but the main focus here was to show how easy it is to build a
clean web-based user interface for our model using Streamlit and deploy it on
Heroku.

References :

[https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku](https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku)
