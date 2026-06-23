import io
import uuid
import time
from pathlib import Path
from fastapi.responses import JSONResponse
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
from fashn_vton import TryOnPipeline
from fastapi import Request
app = FastAPI()
@app.middleware("http")
async def log_total_http_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    total = time.time() - start
    print(f"HTTP_TOTAL {request.url.path}: {total:.3f}s")
    return response

OUTPUT_DIR = Path("api_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = TryOnPipeline(weights_dir="./weights", device="cuda")


def read_image(data: bytes):
    return Image.open(io.BytesIO(data)).convert("RGB")

@app.get("/last")
def last_output():
    files = sorted(OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"error": "no output"}
    return FileResponse(files[0], media_type="image/png")
@app.post("/echo-upload")
async def echo_upload(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    t0 = time.time()
    p = await person_image.read()
    g = await garment_image.read()
    t1 = time.time()
    print(f"ECHO_UPLOAD_READ: {t1-t0:.3f}s, person={len(p)}, garment={len(g)}")
    return {
        "person_size": len(p),
        "garment_size": len(g),
        "read_time": round(t1-t0, 3),
    }

@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(10),
    seed: int = Form(42),
):
    t0 = time.time()

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    t1 = time.time()

    person = read_image(person_bytes)
    garment = read_image(garment_bytes)

    t2 = time.time()

    with torch.inference_mode():
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=num_timesteps,
            guidance_scale=1.5,
            seed=seed,
            segmentation_free=True,
        )

    t3 = time.time()

    out = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
    result.images[0].save(out)

    t4 = time.time()

    print("=" * 80)
    print(f"UPLOAD_READ: {t1 - t0:.3f}s")
    print(f"IMAGE_DECODE: {t2 - t1:.3f}s")
    print(f"PIPELINE_TOTAL: {t3 - t2:.3f}s")
    print(f"SAVE_PNG: {t4 - t3:.3f}s")
    print(f"SERVER_TOTAL: {t4 - t0:.3f}s")
    print(f"PERSON_SIZE: {len(person_bytes)/1024/1024:.2f} MB")
    print(f"GARMENT_SIZE: {len(garment_bytes)/1024/1024:.2f} MB")
    print("=" * 80)

    return FileResponse(out, media_type="image/png")

@app.post("/tryon-json")
async def tryon_json(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(10),
    seed: int = Form(42),
):
    t0 = time.time()

    person = read_image(await person_image.read())
    garment = read_image(await garment_image.read())

    with torch.inference_mode():
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=num_timesteps,
            guidance_scale=1.5,
            seed=seed,
            segmentation_free=True,
        )

    out = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
    result.images[0].save(out)

    total = time.time() - t0
    print(f"TRYON_JSON_TOTAL: {total:.3f}s")

    return JSONResponse({
        "ok": True,
        "server_total": round(total, 3),
        "file": str(out)
    })
@app.get("/files/{filename}")
def get_file(filename: str):
    return FileResponse(OUTPUT_DIR / filename, media_type="image/png")


@app.post("/tryon-link")
async def tryon_link(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(10),
    seed: int = Form(42),
):
    t0 = time.time()

    person = read_image(await person_image.read())
    garment = read_image(await garment_image.read())

    with torch.inference_mode():
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=num_timesteps,
            guidance_scale=1.5,
            seed=seed,
            segmentation_free=True,
        )

    filename = f"{uuid.uuid4().hex}.png"
    out = OUTPUT_DIR / filename
    result.images[0].save(out)

    total = time.time() - t0

    return JSONResponse({
        "ok": True,
        "server_total": round(total, 3),
        "image_path": f"/files/{filename}",
        "image_url": f"https://bfmnn3gx1mrcut-8000.proxy.runpod.net/files/{filename}"
    })
