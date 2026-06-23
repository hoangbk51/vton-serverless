import base64
import io
import os
import time

import runpod
import torch
from PIL import Image
from fashn_vton import TryOnPipeline

print("SERVERLESS HANDLER STARTING", flush=True)

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./weights")

print(f"WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
print("LOADING PIPELINE", flush=True)
pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
print("PIPELINE LOADED", flush=True)


def image_from_base64(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    print("JOB RECEIVED", flush=True)

    t0 = time.time()
    data = job["input"]

    person_image = image_from_base64(data["person_image_base64"])
    garment_image = image_from_base64(data["garment_image_base64"])

    print("RUNNING PIPELINE", flush=True)

    with torch.inference_mode():
        result = pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=data.get("category", "tops"),
            garment_photo_type=data.get("garment_photo_type", "model"),
            num_samples=1,
            num_timesteps=int(data.get("num_timesteps", 10)),
            guidance_scale=1.5,
            seed=int(data.get("seed", 42)),
            segmentation_free=True,
        )

    total = time.time() - t0
    print(f"JOB DONE in {total:.3f}s", flush=True)

    return {
        "ok": True,
        "server_total": round(total, 3),
        "image_base64": image_to_base64(result.images[0]),
    }


runpod.serverless.start({"handler": handler})
