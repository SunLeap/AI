import os, json, joblib, csv, unicodedata, re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

VEC_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
CLS_PATH = os.path.join(DATA_DIR, "model.pkl")
REG_PATH = os.path.join(DATA_DIR, "star_model.pkl")
LABELS_JSON = os.path.join(DATA_DIR, "labels.json")
EMOJI_JSON = os.path.join(DATA_DIR, "emoji_map.json")
SUGG_JSON = os.path.join(DATA_DIR, "suggestions.json")
LOG_CSV = os.path.join(DATA_DIR, "sentiment_records.csv")

vectorizer = None
classifier = None
regressor = None
labels = ["negative","neutral","positive"]
emoji_map = {}
suggestions = []

def _norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\\s+", " ", s).strip()

def _replace_emojis(text, mp):
    if not mp:
        return text
    keys = sorted(mp.keys(), key=len, reverse=True)
    if not keys:
        return text
    pattern = re.compile("|".join(map(re.escape, keys)))
    return pattern.sub(lambda m: " " + mp.get(m.group(0), "") + " ", text)

def _preprocess(text):
    t = _norm(text)
    t = _replace_emojis(t, emoji_map)
    return t

def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _load_artifacts():
    global vectorizer, classifier, regressor, labels, emoji_map, suggestions
    if os.path.exists(VEC_PATH):
        vectorizer = joblib.load(VEC_PATH)
    if os.path.exists(CLS_PATH):
        classifier = joblib.load(CLS_PATH)
    if os.path.exists(REG_PATH):
        regressor = joblib.load(REG_PATH)
    labels = _load_json(LABELS_JSON, labels)
    emoji_map = _load_json(EMOJI_JSON, {})
    suggestions = _load_json(SUGG_JSON, [])

def _classify(text):
    if not (vectorizer and classifier):
        raise RuntimeError("model not loaded")
    X = vectorizer.transform([_preprocess(text)])
    proba = None
    label = classifier.predict(X)[0]
    if hasattr(classifier, "predict_proba"):
        p = classifier.predict_proba(X)[0]
        cls = list(getattr(classifier, "classes_", [])) or labels
        proba = {str(cls[i]): float(p[i]) for i in range(len(p))}
        conf = float(max(p))
    else:
        conf = 1.0
    return label, conf, proba

def _stars(text):
    if not (vectorizer and regressor):
        return None
    X = vectorizer.transform([_preprocess(text)])
    s = float(regressor.predict(X)[0])
    return max(1.0, min(5.0, s))

def _star_from_sentiment(proba):
    if not proba:
        return None
    anchors = {"negative": 1.5, "neutral": 3.0, "positive": 4.5}
    return float(sum(anchors[k] * float(proba.get(k, 0.0)) for k in anchors))

def _fuse_stars(raw_stars, proba, label, confidence):
    anchor = _star_from_sentiment(proba)
    if raw_stars is None:
        return anchor
    fused = 0.7 * float(raw_stars) + 0.3 * (anchor if anchor is not None else 3.0)
    if label == "negative" and float(confidence) >= 0.8:
        fused = min(fused, 3.0)
    if label == "positive" and float(confidence) >= 0.8:
        fused = max(fused, 3.0)
    return float(max(1.0, min(5.0, round(fused, 2))))

def _enough_text(text):
    tokens = _preprocess(text).split()
    return len(tokens) >= 6

def _pick_suggestions(text, label, stars):
    # only show suggestions for clearly negative cases
    if not (label == "negative" or (stars is not None and stars <= 3.0)):
        return []

    out = []
    t = _preprocess(text).lower()

    for row in suggestions:
        kws = [k.lower() for k in row.get("keywords", [])]
        if any(k and k in t for k in kws):
            msg = row.get("suggested_action") or row.get("topic")
            if msg and msg not in out:
                out.append(msg)

    if not out:
        out.append("Acknowledge the issue, apologize, explain next steps, and offer a make-good.")
    return out[:6]


def _log_prediction(text, label, confidence, stars):
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","text","label","confidence","stars"])
        w.writerow([datetime.utcnow().isoformat(), text, label, f"{confidence:.4f}", "" if stars is None else f"{stars:.2f}"])

@app.route("/")
def index():
    if os.path.exists(os.path.join(TEMPLATES_DIR, "index.html")):
        return render_template("index.html")
    return jsonify({"ok": True, "message": "index.html not found. Place it under /templates"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=False) or {}
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error":"Empty text"}), 400

        label, conf, proba = _classify(text)
        raw_stars = _stars(text) if _enough_text(text) else None
        final_stars = _fuse_stars(raw_stars, proba, label, conf)
        suggs = _pick_suggestions(text, label, final_stars)
        _log_prediction(text, label, conf, final_stars if final_stars is not None else raw_stars)

        return jsonify({
            "label": label,
            "confidence": conf,
            "proba": proba,
            "stars": final_stars,
            "suggestions": suggs
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "has_vectorizer": bool(vectorizer),
        "has_classifier": bool(classifier),
        "has_regressor": bool(regressor),
        "labels": labels
    })

@app.route("/api/info")
def api_info():
    return jsonify({
        "labels": labels,
        "uses_tfidf": True,
        "classifier": "Calibrated LinearSVC",
        "regressor": "LinearSVR" if regressor else None,
        "emoji_expansion": bool(emoji_map),
        "suggestions_rules": len(suggestions)
    })

@app.route("/static/<path:filename>")
def static_passthrough(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    _load_artifacts()
    app.run(host="0.0.0.0", port=8000, debug=True)
