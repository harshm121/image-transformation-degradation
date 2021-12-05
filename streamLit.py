import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from torchvision import transforms
from torchvision.models import vgg16, resnet152, alexnet
import torch

import os
import json
from PIL import Image
import random

softmax = torch.nn.Softmax(dim = 1)
st.set_page_config(page_title='ImageNet performance degradation')
col1, col2 = st.columns(2)

imageNames = {"Bulldog": ['ILSVRC2012_val_00027433'],
              "Cat": ['ILSVRC2012_val_00045239'],
             "Wolf": ['ILSVRC2012_val_00001172'],
             "Teapot": ['ILSVRC2012_val_00006001']}

imageNetClassIdx = {"Bulldog": 245, "Cat": 281, "Wolf": 269, 'Teapot': 849}

transformationToFolder = {'None': "original",
                          "Shuffle": "shuffle_quad",
                          'Rotate by 45': "rotate",
                          'Vertical Shift':"vert",
                          'Horizontal shift':"horizontal",
                          'Hue + 30': 'hueP30'}

imageNetClass = st.sidebar.selectbox("ImageNet Class", ["Teapot", "Wolf", "Cat", "Bulldog"])

transformation = st.sidebar.selectbox("Transformation", ['Shuffle', 'Rotate by 45', 'Vertical Shift', 'Horizontal shift'])

st.sidebar.write('Run Models:')
model_options = ['Alexnet', 'ResNet152', 'VGG16']
model_option_checkboxes = []
for model_option in model_options:
    model_option_checkboxes.append(st.sidebar.checkbox(model_option))
open_models = [model_options[i] for i, checked in enumerate(model_option_checkboxes) if checked]

imgName = random.choice(imageNames[imageNetClass]) + ".JPEG"
image_path = os.path.join(os.path.join("./streamLitHelp/images", transformationToFolder["None"]), imgName)
groundTruthClass = imageNetClassIdx[imageNetClass]

preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

img = Image.open(image_path).convert('RGB')
img_processed = preprocessing(img).unsqueeze(dim = 0)

imageT_path = os.path.join(os.path.join("./streamLitHelp/images", transformationToFolder[transformation]), imgName)
imgT = Image.open(imageT_path).convert('RGB')
imgT_processed = preprocessing(imgT).unsqueeze(dim = 0)

@st.cache
def computeScore(model_name, target_img):
    if(model_name == "ResNet152"):
        model = resnet152(pretrained = True)
    elif(model_name == "VGG16"):
        model = vgg16(pretrained = True)
    elif(model_name == "Alexnet"):
        model = alexnet(pretrained = True)

    model.eval()
    return model(target_img)

def renderImages():
    col1.header("Original")
    col1.image(img, caption=(imageNetClass + ", Original"), width = 300)
    col2.header("Transformed")
    col2.image(imgT, caption=(imageNetClass + ", " + transformation), width = 300)

    scores = []
    scoresT = []

    for model_name in model_options:
        probClass = np.nan
        probTClass = np.nan
        if model_name in open_models:
            out = softmax(computeScore(model_name, img_processed)).detach().numpy()[0]
            outT = softmax(computeScore(model_name, imgT_processed)).detach().numpy()[0]
            probClass = out[groundTruthClass]
            probTClass = outT[groundTruthClass]
            # if(probClass<0.001):
            #     probClass = "{:.3e}".format(probClass)
            # else:
            #     probClass = "{:.4f}".format(probClass)
            # if(probTClass<0.001):
            #     probTClass = "{:.3e}".format(probTClass)
            # else:
            #     probTClass = "{:.4f}".format(probTClass)
        scores.append(probClass)
        scoresT.append(probTClass)
    P_label = "P({})".format(imageNetClass)
    with col1:
        st.table(pd.DataFrame(
            data=scores,
            index=model_options,
            columns=[P_label]
        ).style.format({P_label: '{:.4f}'}))
    with col2:
        st.table(pd.DataFrame(
            data=np.asarray([scoresT, 1-np.divide(scoresT, scores)]).T,
            index=model_options,
            columns=[P_label, "Damage"]
        ).style.format({P_label: '{:.4f}', "Damage": '{:.4f}'}))


# print(modelName, transformation, imageNetClass)
renderImages()
# renderScores()
