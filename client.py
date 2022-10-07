from statistics import mode
import torch
import torch.nn as nn
from torch import tensor
import torchvision
import torchvision.transforms as transforms
import numpy as np
from shoe_classifier import *
import PIL
import streamlit as st
from PIL import Image




model1 = "shoeclassifier.pt"

model2 = torch.load(model1)
def emotion(img):
    images = PIL.Image.open(img)
    trans = transforms.ToPILImage()
    trans1=transforms.ToTensor()
    img_req = (trans1(images))
    plt.imshow(trans(trans1(images)))
    model2.eval()
    rgb_im = images.convert('RGB')
    img = tr(rgb_im)
    img = img.unsqueeze(dim=0)
    img = img.to(device)
    print(model2(img))
    max = torch.argmax(model2(img))
    item_labels = ['adidas','nike']
    print(f'Predicted image is {item_labels[max]}')
    return model2(img)

def emotion2(img):
    images = PIL.Image.open(img)
    trans = transforms.ToPILImage()
    trans1=transforms.ToTensor()
    img_req = (trans1(images))
    plt.imshow(trans(trans1(images)))
    model2.eval()
    rgb_im = images.convert('RGB')
    img = tr(rgb_im)
    img = img.unsqueeze(dim=0)
    img = img.to(device)
    print(model2(img))
    max = torch.argmax(model2(img))
    item_labels = ['adidas','nike']
    print(f'Predicted image is {item_labels[max]}')
    return max

# k=emotion("D:/neural networks/cifar10/1412-lpt-1614146683.jpg")
# na = k.detach().to('cpu').numpy()
# print(na)



st.title("Shoe Classifier ")
file_up = st.file_uploader("Upload an image", type=["jpeg","jpg","png","webp"])
def predict(image):
    
    vals = emotion2(image)
    v1 = emotion(image)
    na = v1.detach().to('cpu').numpy()
    # st.write(na)
    # st.write(v1)
    
    classes = ['adidas','nike']

    # return the prediction ranked by highest probabilities
    
    return [classes[vals],np.amax(na)]

if file_up is not None:
    # displays uploaded image
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)
    #st.write(labels)

    # print out the predictions with scores, highest probability value is the most likely case
    
    st.write("Prediction (index, name)", labels[0], ",   Score: ",labels[1])



