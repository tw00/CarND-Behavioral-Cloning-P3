import argparse
import base64
import json
import pickle
import random

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import yuv_colorspace
import cv2
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
#prev_image_array = None
last_steering = 0;

def preprocess_img(img):
    IMG_W = 128
    IMG_H = 64

    #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img = img.convert('RGB')
    img = np.array(img)
    #print(img);
    #print(img.shape);
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV does not use RGB, it uses BGR
    img = cv2.resize(img, (IMG_W, IMG_H))

    img = img.astype(float)/255.0
    img = yuv_colorspace.rgb2yuv(img) # convert to YUV colorspace
    img[:,:,0] = img[:,:,0] - 0.5; # remove mean
    return img

@sio.on('telemetry')
def telemetry(sid, data, alpha=2.5):
    global last_steering
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    print("current state: sa = {}, f = {}, s = {}: ".format(steering_angle, throttle, speed), end="");
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = preprocess_img( image );
    image_array = np.asarray(image)
    #print(image_array.shape);
    transformed_image_array = image_array[None, :, :, :]
    if (random.random() < 0.1):
        with open("foo.p", "wb") as f:
            pickle.dump(transformed_image_array, f)
    #print(transformed_image_array.shape);
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #
    #print("calling model");
    #print(transformed_image_array.shape);
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #steering_angle = 0.1
    new_steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = (1-1/alpha)*last_steering + (1/alpha)*new_steering_angle
    last_steering = steering_angle;
    #steering_angle = steering_angle * 5;
    throttle = 0.05
    print("predicted sa = {}, throttle = {}".format(steering_angle, throttle))
    #print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
