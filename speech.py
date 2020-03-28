from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from bangla_tts import generate
import os

os.system('python opencv_text_detection_image.py --image images/Book2.jpg --east frozen_east_text_detection.pb')

model = load_learner("Model", "model.pkl")

text = ""

path = "output/"
for i in range(1,5):
    path_img = open_image(Path(path+str(i)+'.jpg'))
    result = model.predict(path_img)
    text = text + " " + str(result[0])

print(text)

file_names = generate([text], save_path = "static") # will be saved to static folder
print(file_names)
