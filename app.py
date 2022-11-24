import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.title('Image classification')
st.write('Fine tuned EfficientNet V2 small variation for just 1 epoch')
st.write('Validation set showed about 90%')
st.header('Upload your image')

image = st.file_uploader('Upload')

model = torch.load('model_effnet.pt', map_location=torch.device('cpu'))

transform = transforms.Compose([
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])

class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def check_test(image, transform, model, class_names):
    image = Image.open(image)
    tr_image = transform(image).view((1, 3, 224, 224))
    y_pred = model(tr_image)
    lst = torch.softmax(y_pred, dim=1).detach().numpy().flatten()
    lst = list(map(lambda x: np.round(100*x, 2), lst))
    # print(len(lst))
    preds = pd.DataFrame({'class': class_names, 'probability': lst}).sort_values(by='probability', ascending=False)
    preds['probability'] = preds['probability'].apply(lambda x: f"{x} %")

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.dataframe(preds)

if image:
    check_test(image, transform, model, class_names)

else:
    st.write('I am waiting for image')
