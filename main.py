import os
import torch
import uvicorn
import gradio as gr
from PIL import Image
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response



# MODEL_DIR = os.environ["MODEL_DIR"]
# BASE_DIR = os.environ["BASE_DIR"]

BASE_DIR = 'ultralytics/yolov5'
MODEL_DIR = 'best.pt'

model = torch.hub.load(BASE_DIR, 'custom', path=MODEL_DIR, device='cpu', trust_repo=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.glissai.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get('/')
def root():
    return {"message": "hello!"}


class_category={'picture': 0, 'sofa': 1, 'both': None}

def gradio_infer(input_image, key, conf):
     if input_image:
         model.classes = class_category[key]
         model.conf = conf
         gresults = model(input_image, size=640) 
         gresults.render()
         gres=Image.new(mode="RGB", size=(1, 1)) # blank img
         for img in gresults.ims:
              bytes_io = BytesIO()
              gres = Image.fromarray(img)
         return gres


io = gr.Interface(fn=gradio_infer, 
     inputs=[gr.Image(type="pil"), 
            gr.Radio(["picture", "sofa", "both"], value='both', label="What object do you want to detect?"), 
            gr.Slider(0.50, 0.90, step=0.10, value=0.70, label="Set the confidence level:")],
     outputs=gr.Image(type="pil"),
     examples=[["testimg.jpg", "picture", 0.8], ["testimg2.jpg", "sofa", 0.7]],
     allow_flagging = 'never',
     css="footer {visibility: hidden}",
     live=True
     )

gr.mount_gradio_app(app, io, path="/gradio")


if __name__ == "__main__":

     uvicorn.run(app, host='0.0.0.0', port=8080)
