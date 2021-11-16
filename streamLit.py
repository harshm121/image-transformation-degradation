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

modelName = st.sidebar.radio("Model", ['ResNet152', 'Alexnet', 'VGG16'])
    
transformation = st.sidebar.selectbox("Transformation", ['Shuffle', 'Rotate by 45', 'Vertical Shift', 'Horizontal shift'])

imageNetClass = st.sidebar.selectbox("ImageNet Class", ["Teapot", "Wolf", "Cat", "Bulldog"])

# print(modelName, transformation, imageNetClass)

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

if(modelName == "ResNet152"):
    model = resnet152(pretrained = True)
elif(modelName == "VGG16"):
    model = vgg16(pretrained = True)
elif(modelName == "Alexnet"):
    model = alexnet(pretrained = True)
    
model.eval()
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
out = softmax(model(img_processed)).detach().numpy()[0]
probClass = out[groundTruthClass]

st.image(img, caption='Original Image', width = 300)
st.markdown("{}'s Probability of {} = {}".format(modelName, imageNetClass, str(probClass)))

imageT_path = os.path.join(os.path.join("./streamLitHelp/images", transformationToFolder[transformation]), imgName)


imgT = Image.open(imageT_path).convert('RGB')
imgT_processed = preprocessing(imgT).unsqueeze(dim = 0)
outT = softmax(model(imgT_processed)).detach().numpy()[0]
probTClass = outT[groundTruthClass]

st.image(imgT, caption='Transformed Image', width = 300)
st.markdown("{}'s Probability of transformed {} = {}".format(modelName, imageNetClass, str(probTClass)))
                          