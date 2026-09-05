from pathlib import Path
import json
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_chest_xray_model.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.46
UNCERTAINTY_MARGIN = 0.10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
model_config_path = BASE_DIR / "model_config.json"
if model_config_path.exists():
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    candidate_model_path = BASE_DIR / model_config["model_path"]
    if candidate_model_path.exists():
        MODEL_PATH = candidate_model_path
        THRESHOLD = float(model_config["threshold"])
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    probability = float(model.predict(tf.expand_dims(image_array, axis=0), verbose=0)[0][0])

    if abs(probability - THRESHOLD) < UNCERTAINTY_MARGIN:
        label = "Uncertain"
        detail = "النتيجة قريبة من الحد الفاصل وتحتاج إلى مراجعة اختصاصي أشعة."
        tone = "uncertain"
    elif probability >= THRESHOLD:
        label = "Pneumonia"
        detail = "النموذج رصد نمطًا متوافقًا مع الالتهاب الرئوي."
        tone = "warning"
    else:
        label = "Normal"
        detail = "النموذج لم يرصد نمطًا واضحًا متوافقًا مع الالتهاب الرئوي."
        tone = "healthy"

    return {
        "label": label,
        "detail": detail,
        "tone": tone,
        "probability": round(probability * 100, 1),
        "normal_probability": round((1 - probability) * 100, 1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    preview_name = None

    if request.method == "POST":
        uploaded_file = request.files.get("xray")
        if not uploaded_file or not uploaded_file.filename:
            error = "اختار صورة أشعة أولًا."
        elif not allowed_file(uploaded_file.filename):
            error = "الملفات المقبولة: PNG أو JPG أو JPEG أو WEBP."
        else:
            upload_dir = BASE_DIR / "uploads"
            upload_dir.mkdir(exist_ok=True)
            extension = Path(secure_filename(uploaded_file.filename)).suffix.lower()
            saved_name = f"{uuid.uuid4().hex}{extension}"
            saved_path = upload_dir / saved_name
            uploaded_file.save(saved_path)
            preview_name = saved_name
            try:
                result = predict_image(saved_path)
            except (ValueError, OSError):
                error = "تعذر قراءة الصورة. جرّب ملف أشعة بصيغة مختلفة."

    return render_template("index.html", result=result, error=error, preview_name=preview_name)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory

    return send_from_directory(BASE_DIR / "uploads", filename)


if __name__ == "__main__":
    app.run(debug=True)