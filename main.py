import uvicorn;
from fastapi import FastAPI, File, UploadFile;
from pydantic import BaseModel;
from typing import Optional;
import os;

import tensorflow as tf;
from keras import backend as K;

import numpy as np;
import matplotlib.pyplot as plt;
import cv2;
from PIL import Image;

app = FastAPI();

def f1score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model = tf.keras.models.load_model('../Thomas/Fourier_Xception_Binary', custom_objects={'f1score':f1score})

@app.get("/")
def index(b: bool):
    return b;

class Model(BaseModel):
    rattr: float = 5.9;
    attr: Optional[str];
@app.post("/")
def blog(req: Model):
    return req;

@app.post("/img")
async def pred(img: bytes = File()):
    return {
        "img_size_byte": len(img),
    };

@app.post("/upimg/")
async def create_upload_file(file: UploadFile):
    file_str = await file.read();
    np_img = np.fromstring(file_str, np.uint8);
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE);
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.imsave("./img/fake/"+file.filename, magnitude_spectrum);

    # PIL_image = Image.fromarray(np.uint8(magnitude_spectrum)).convert('RGB').resize((299, 299))
    # plt.imshow(PIL_image)
    # plt.show();

    val = tf.keras.utils.image_dataset_from_directory(
        "./img/",
        seed=1337,
        image_size=(299, 299),
        batch_size=32,
    );
    pred = model.predict(val);

    os.replace("./img/fake/"+file.filename, "./history/"+ file.filename);

    return {
        "filename": file.filename,
        "pred": pred.tolist(),
    };

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=6620, reload=True);