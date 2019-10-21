import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image


#dict
emotions_dict = {'angry':0,
                'sad':1,
                'surprised':2,
                'happy':3}

def generate_df(dir, images, list_emotions):
    emotions  = ['angry','sad','surprised','happy']
    for emotion in emotions:
        IMG_PATH = "/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/"+dir+"/"+emotion
        for filename in os.listdir(IMG_PATH):
            if filename.endswith(".jpeg"):
                img = Image.open(IMG_PATH+"/"+filename).convert('L')  # convert image to 8-bit grayscale
                data = list(img.getdata())
                df = pd.DataFrame([data])
                list_emotions.append(emotion)
                images = images.append(df)
    return images, list_emotions


images = pd.DataFrame()       
emotion_list = []
images_df, list_emotions = generate_df('modification-kopia', images, emotion_list)
images_df['img'] = images_df[images_df.columns].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
images = images_df[['img']]
images['emotion'] = list_emotions
images['emotion_id'] = images.emotion.apply(lambda x: emotions_dict[x])

images.to_csv('images.csv', index=True, header=True)

