# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint
import requests
import gradio as gr
from io import BytesIO
from PIL import Image

# DETECTION_URL = "a.run.app/obj_detect"
DETECTION_URL = "http://127.0.0.1:8080/obj_detect"

IMAGE = "testimg.jpg"

def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()

def run_api(img):
	# Read image
	# f = open(IMAGE, 'rb')
	# files = {"my_file": (f.name, f)}
	inimg = convertToJpeg(img)
	files = {"my_file":inimg}
	print("file ready")
	response = requests.post(DETECTION_URL, files=files)
	
	# pprint.pprint(response)
	image = Image.open(BytesIO(response.content))
	print('image made')
	return image

def run_gradio():
	return gr.Interface(fn=run_api, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Image(type="pil"),
             examples=[IMAGE]).launch()

if __name__ == "__main__":
	run_gradio()