from flask import Flask, jsonify, request
import io
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os

app = Flask(__name__)

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mineral_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 11, stride=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 1), #out 70x70

            nn.Conv2d(48, 128, 5, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 1),#out 64x64

            nn.Conv2d(128, 128, 4, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(4, 3),#out 20x20

            nn.Conv2d(128, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 3),#out 20x20

            nn.Flatten(),
            nn.Linear(64*6*6, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),
            nn.Linear(512, 7),
            nn.LogSoftmax(dim=1),
            )

    def forward(self, x):
        out = self.net(x)
        return out


model = Mineral_1().to(device)
model.load_state_dict(torch.load('weights/final.h5',map_location=torch.device('cpu')))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)





class_dict = {
0: 'biotite',
1: 'bornite',
2: 'chrysocolla',
3: 'malachite',
4: 'muscovite',
5: 'pyrite',
6: 'quartz',
}

def get_prediction(image_bytes):
    prediction_list = []
    tensor = transform_image(image_bytes = image_bytes).to(device)
    out = model(tensor)
    ps = torch.exp(out)
    _, top_class = torch.max(ps , 1)
    preds = np.squeeze(top_class.cpu().numpy())
    #print(f"type: {type(preds)}")
    #print(f"size: {preds.size}")
    prediction_list.append(preds)
    #print("predicton shape {}".format(prediction.shape))
    return np.squeeze(prediction_list), tensor.shape



@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction, image_shape = get_prediction(image_bytes=img_bytes)
        class_name = str(class_dict[int(prediction)])
        return jsonify({'Class_Name': class_name})


if __name__ == '__main__':
    app.run()
