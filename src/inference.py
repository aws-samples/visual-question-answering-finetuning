import base64
import logging
import sys
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
LOGGER = logging.Logger("Inference-BLIP2", level=logging.DEBUG)
HANDLER = logging.StreamHandler(sys.stdout)
HANDLER.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(HANDLER)


def model_fn(model_dir):
    # load model and processor from model_dir
    LOGGER.log(logging.DEBUG, ("Loading model and processor"))
    model = Blip2ForConditionalGeneration.from_pretrained(model_dir, device_map="auto", load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def predict_fn(data, model_and_processor):
    # unpack model and processor
    LOGGER.log(logging.DEBUG, ("Received Data"))
    model, processor = model_and_processor
    # preprocess
    base64_image_string = data.pop("image")
    f = BytesIO(base64.b64decode(base64_image_string))
    image = Image.open(f).convert("RGB")

    if "prompt" in data:
        prompt = data.pop("prompt")
    else:
        LOGGER.log(logging.DEBUG, ("No Prompt Received"))
        prompt = None

    if "parameters" in data:
        params = data.pop("parameters")
    else:
        LOGGER.log(logging.DEBUG, ("No parameters received, using default"))
        params = {}
    LOGGER.log(logging.DEBUG, (f"Inputs: {prompt}, {params}"))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    output = model.generate(**inputs, **params)

    generated_text = processor.decode(output[0], skip_special_tokens=True)
    return generated_text
