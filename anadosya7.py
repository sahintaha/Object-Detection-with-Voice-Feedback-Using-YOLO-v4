#-*- coding: utf-8 -*-  #chars

import cv2
from pygame import mixer
import time
from gtts import gTTS
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime
import os

def seslendir_thread(class_name):
    metin = "Bir " + class_name + " algılandı."
    tts = gTTS(metin, lang="tr")
    now = datetime.datetime.now()
    dosya_adi = str(class_name)+str(time.time())+'.mp3'
    if os.path.isfile(dosya_adi):
        os.remove(dosya_adi)
    tts.save(dosya_adi)
    mixer.init()
    mixer.music.load(dosya_adi)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)
    mixer.quit()
    os.remove(dosya_adi)

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r", encoding="utf-8") as file_object:
    for class_name in file_object.readlines():

        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0)

# Create a set to keep track of already detected objects
detected_objects = set()

while True:

    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)

        class_name = classes[class_id]

        font = ImageFont.truetype("arial.ttf", size=40)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), class_name, font=font, fill=(25, 20, 250, 0))
        frame = np.array(img)

        def bekle_thread():
            while True:
                time.sleep(2)

        start_time = time.time()
        other_object_detected = []

        if class_name not in detected_objects:
            detected_objects.add(class_name)
            other_object_detected.append(class_name)
            threading.Thread(target=seslendir_thread, args=(class_name,)).start()
            threading.Thread(target=time.sleep, args=(3,)).start()

        key = cv2.waitKey(1)

        def timer(duration):
            time.sleep(duration)
            print("tamam")
            detected_objects.remove(class_name)

        if other_object_detected:
            threading.Thread(target=timer, args=(5,)).start()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

