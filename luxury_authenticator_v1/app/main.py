from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference.predict import Predictor


app = FastAPI(
    title="LuxLoc Luxury Authenticator",
    description="AI-assisted authenticity assessment",
)

predictor = Predictor("models/luxury_authenticator.pth")

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def home():
    index_file = Path("frontend/index.html")

    if not index_file.exists():
        return "<h1>LuxLoc Authenticator</h1>"

    return index_file.read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/avif",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format",
        )

    suffix = Path(file.filename or "image.jpg").suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = Path(tmp.name)

    try:
        image = Image.open(temp_path)
        return predictor.predict(image)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )

    finally:
        temp_path.unlink(missing_ok=True)