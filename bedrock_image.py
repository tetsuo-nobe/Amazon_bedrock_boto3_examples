import boto3
import json
import base64
import io
import os
import sys
from PIL import Image

bedrock = boto3.client('bedrock-runtime')

accept = 'application/json'
contentType = 'application/json'

modelId1="stability.stable-diffusion-xl"

# Text to Image

prompt = "a beautiful mountain landscape"
negative_prompts = [
    "poorly rendered",
    "poor background details",
    "poorly drawn mountains",
    "disfigured mountain features",
]
style_preset = "photographic"  # (e.g. photographic, digital-art, cinematic, ...)


# リクエストBODYの指定
body_stability_stable_diffusion_xl = json.dumps({
    "text_prompts": (
        [{"text": prompt, "weight": 1.0}]
        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
    ),
    "cfg_scale": 5,
    "seed": 5450,
    "steps": 70,
    "style_preset": style_preset,
})


response = bedrock.invoke_model(body=body_stability_stable_diffusion_xl,
                                modelId=modelId1
                                )
                                
response_body = json.loads(response.get("body").read())

print(response_body["result"])
base_64_img_str = response_body["artifacts"][0].get("base64")
print(f"{base_64_img_str[0:80]}...")

os.makedirs("data", exist_ok=True)
image_1 = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
image_1.save("data/image_1.png")

# Image to Image

def image_to_base64(img) -> str:
    """Convert a PIL Image or local image file path to a base64 string for Amazon Bedrock"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        print("Converting PIL Image to base64 string")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")


init_image_b64 = image_to_base64(image_1)
print(init_image_b64[:80] + "...")

change_prompt = "add some trees in the background"

request = json.dumps({
    "text_prompts": (
        [{"text": change_prompt, "weight": 1.0}]
        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
    ),
    "cfg_scale": 10,
    "init_image": init_image_b64,
    "seed": 321,
    "start_schedule": 0.6,
    "steps": 50,
    "style_preset": style_preset,
})
modelId = "stability.stable-diffusion-xl"

response = bedrock.invoke_model(body=request, modelId=modelId1)
response_body = json.loads(response.get("body").read())

print(response_body["result"])
image_2_b64_str = response_body["artifacts"][0].get("base64")
print(f"{image_2_b64_str[0:80]}...")

image_2 = Image.open(io.BytesIO(base64.decodebytes(bytes(image_2_b64_str, "utf-8"))))
image_2.save("data/image_2.png")

# Image Inpainting

os.makedirs("data", exist_ok=True)
mask = Image.open('data/mask.png')

inpaint_prompt = "add a forest cabin"

request = json.dumps({
    "text_prompts":[{"text": inpaint_prompt}],
    "init_image": image_to_base64(image_2),
    "mask_source": "MASK_IMAGE_BLACK",
    "mask_image": image_to_base64(mask),
    "cfg_scale": 10,
    "seed": 321,
    "style_preset": style_preset,
})

response = bedrock.invoke_model(body=request, modelId=modelId1)
response_body = json.loads(response.get("body").read())

print(response_body["result"])
image_3_b64_str = response_body["artifacts"][0].get("base64")
print(f"{image_2_b64_str[0:80]}...")

os.makedirs("data", exist_ok=True)
inpaint = Image.open(io.BytesIO(base64.decodebytes(bytes(image_3_b64_str, "utf-8"))))
inpaint.save("data/inpaint.png")