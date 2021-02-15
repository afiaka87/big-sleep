from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run()

import io
from base64 import encodebytes
from PIL import Image


# from flask import jsonify

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


# server side code
image_path = 'images/test.png'  # point to your image location
encoded_img = get_response_image(image_path)
my_message = 'here is my message'  # create your message as per your need
response = {'Status': 'Success', 'message': my_message, 'ImageBytes': encoded_img}
# return jsonify(response) # send the result to client
