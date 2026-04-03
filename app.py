from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app)

# Lazy load model (prevents startup crash)
model = None

CLASS_NAMES = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            "alzheimer_model.h5",
            compile=False
        )
    return model


@app.route("/modelinfo", methods=["GET"])
def modelinfo():
    try:
        model = get_model()
        return jsonify({
            "input_shape":  str(model.input_shape),
            "output_shape": str(model.output_shape),
            "model_type":   type(model).__name__,
            "classes":      CLASS_NAMES,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate files
        if "mri" not in request.files or "cha" not in request.files:
            return jsonify({"error": "Both 'mri' and 'cha' files are required"}), 400

        mri_file = request.files["mri"]
        cha_file = request.files["cha"]

        # ───── MRI Processing ─────
        img = Image.open(io.BytesIO(mri_file.read())).convert("L")
        img = img.resize((180, 180))

        mri_arr = np.array(img, dtype=np.float32) / 255.0
        mri_arr = mri_arr.reshape(1, 180, 180, 1)

        model = get_model()
        prediction = model.predict(mri_arr)

        class_idx  = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])
        class_name = CLASS_NAMES[class_idx]
        label      = "Healthy" if class_idx == 0 else "MCI/AD"

        # ───── CHA Processing ─────
        cha_text = cha_file.read().decode("utf-8")
        features = extract_features(cha_text)

        return jsonify({
            "label": label,
            "class": class_name,
            "confidence": round(confidence * 100, 2),
            "all_scores": {
                CLASS_NAMES[i]: round(float(prediction[0][i]) * 100, 2)
                for i in range(4)
            },
            "speech_features": features
        })

    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# FEATURE EXTRACTION (UNCHANGED CORE LOGIC)
# ─────────────────────────────────────────────

def parse_wor_line(wor_line):
    words = []
    parts = wor_line.replace("%wor:", "").strip().split()

    k = 0
    while k < len(parts):
        word = parts[k]

        if k + 1 < len(parts):
            timing = parts[k + 1].replace("\x15", "").strip()

            if "_" in timing and timing.replace("_", "").isdigit():
                try:
                    start_ms, end_ms = map(float, timing.split("_"))

                    words.append({
                        "word": word,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "start_sec": start_ms / 1000.0,
                        "end_sec": end_ms / 1000.0,
                        "duration_sec": (end_ms - start_ms) / 1000.0
                    })

                    k += 2
                    continue
                except:
                    pass

        k += 1

    return words


def extract_features(cha_text):
    lines = cha_text.split("\n")

    all_word_segments = []
    par_data = []
    all_silences = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        if line.startswith("*PAR:"):
            par_words = []

            j = i + 1
            while j < len(lines) and j < i + 10:
                next_line = lines[j].rstrip("\n")

                if next_line.startswith("%wor:"):
                    par_words = parse_wor_line(next_line)
                    break
                elif next_line.startswith("*"):
                    break

                j += 1

            if par_words:
                all_word_segments.extend(par_words)

                par_silence_total = 0.0
                par_speech_total = 0.0
                par_num_silences = 0

                for idx in range(len(par_words) - 1):
                    gap = par_words[idx + 1]["start_sec"] - par_words[idx]["end_sec"]

                    if gap > 0:
                        all_silences.append(gap)
                        par_silence_total += gap
                        par_num_silences += 1

                for w in par_words:
                    par_speech_total += w["duration_sec"]

                par_total_duration = (
                    par_words[-1]["end_sec"] - par_words[0]["start_sec"]
                )

                par_data.append({
                    "total_duration_sec": par_total_duration,
                    "total_speech_sec": par_speech_total,
                    "total_silence_sec": par_silence_total,
                    "num_silences": par_num_silences,
                    "num_words": len(par_words),
                })

        i += 1

    if not all_word_segments:
        return {
            "pause_count": 0,
            "total_speech_time": 0,
            "total_pause_time": 0,
            "mean_word_duration": 0,
            "speech_rate_wpm": 0,
            "pause_per_word": 0,
            "total_words": 0,
        }

    pause_count = sum(p["num_silences"] for p in par_data)
    total_speech_time = sum(p["total_duration_sec"] for p in par_data)
    total_pause_time = sum(p["total_silence_sec"] for p in par_data)
    total_speech_secs = sum(p["total_speech_sec"] for p in par_data)
    total_words = len(all_word_segments)

    mean_word_duration = round(total_speech_secs / total_words, 4) if total_words else 0
    speech_rate_wpm = round((total_words / total_speech_secs) * 60, 2) if total_speech_secs > 0 else 0
    pause_per_word = round(len(all_silences) / total_words, 4) if total_words else 0

    return {
        "pause_count": pause_count,
        "total_speech_time": round(total_speech_time, 2),
        "total_pause_time": round(total_pause_time, 2),
        "mean_word_duration": mean_word_duration,
        "speech_rate_wpm": speech_rate_wpm,
        "pause_per_word": pause_per_word,
        "total_words": total_words,
    }


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)