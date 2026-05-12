
import os
import io
import csv
import json
from datetime import datetime
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, send_file, flash)

import model_utils as mu

# ---------------------------------------------------------------------------
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(APP_DIR, "uploads")
DATA_DIR    = os.path.join(APP_DIR, "static", "data")
FEEDBACK_FP = os.path.join(DATA_DIR, "feedback.csv")
HISTORY_FP  = os.path.join(DATA_DIR, "history.csv")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

app = Flask(__name__)
app.secret_key = "botanical-atlas-snn-crop-dashboard"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ---------------------------------------------------------------------------
# Crop dossier — short note shown on the result card for each crop class
# ---------------------------------------------------------------------------
CROP_DOSSIER = {
    "rice":        {"family": "Cereal",   "season": "Kharif",     "note": "Loves standing water and warm humid air."},
    "maize":       {"family": "Cereal",   "season": "Kharif/Rabi","note": "Versatile staple — drains well, full sun."},
    "chickpea":    {"family": "Pulse",    "season": "Rabi",       "note": "Cool dry winters; nitrogen-fixing legume."},
    "kidneybeans": {"family": "Pulse",    "season": "Kharif",     "note": "Mild summers, well-drained loamy soil."},
    "pigeonpeas":  {"family": "Pulse",    "season": "Kharif",     "note": "Drought-hardy; deep tap-rooted legume."},
    "mothbeans":   {"family": "Pulse",    "season": "Kharif",     "note": "Arid-zone legume — survives sparse rain."},
    "mungbean":    {"family": "Pulse",    "season": "Kharif",     "note": "Short cycle, warm season, sandy loam."},
    "blackgram":   {"family": "Pulse",    "season": "Kharif",     "note": "Warm humid climate, clay-loam soils."},
    "lentil":      {"family": "Pulse",    "season": "Rabi",       "note": "Cool dry; highly nutritious legume."},
    "pomegranate": {"family": "Fruit",    "season": "Perennial",  "note": "Semi-arid, hot dry summers, mild winters."},
    "banana":      {"family": "Fruit",    "season": "Perennial",  "note": "Tropical heat, deep loam, plenty of water."},
    "mango":       {"family": "Fruit",    "season": "Perennial",  "note": "Hot dry pre-flower, then warm humid."},
    "grapes":      {"family": "Fruit",    "season": "Perennial",  "note": "Mediterranean — dry summers, cool winters."},
    "watermelon":  {"family": "Cucurbit", "season": "Summer",     "note": "Hot, sandy, deep watering, full sun."},
    "muskmelon":   {"family": "Cucurbit", "season": "Summer",     "note": "Warm dry — sandy loam, full sun."},
    "apple":       {"family": "Fruit",    "season": "Temperate",  "note": "Cold winters for chill-hours; mild summers."},
    "orange":      {"family": "Citrus",   "season": "Perennial",  "note": "Subtropical, well-drained, frost-free."},
    "papaya":      {"family": "Fruit",    "season": "Perennial",  "note": "Tropical heat, no frost, deep rich soil."},
    "coconut":     {"family": "Palm",     "season": "Perennial",  "note": "Coastal humid tropics, sandy soils."},
    "cotton":      {"family": "Fibre",    "season": "Kharif",     "note": "Warm season, black cotton soils."},
    "jute":        {"family": "Fibre",    "season": "Kharif",     "note": "Hot humid, alluvial soils, monsoon."},
    "coffee":      {"family": "Beverage", "season": "Perennial",  "note": "Highland tropics, shade, well-drained."},
    "tea":         {"family": "Beverage", "season": "Perennial",  "note": "Cool humid hills, acidic loam, rainfall."},
}


def dossier_for(label: str) -> dict:
    return CROP_DOSSIER.get(label.lower(),
                            {"family": "—", "season": "—",
                             "note": "Suitable for the given soil & climate profile."})


# ---------------------------------------------------------------------------
# Agronomic range benchmarks used for auto-feedback sanity checks
# ---------------------------------------------------------------------------
_AGRO_RANGES = {
    # (feature, min_ok, max_ok)  — gross out-of-range triggers a warning note
    "N":           (0,   150),
    "P":           (0,   150),
    "K":           (0,   210),
    "temperature": (8,   45),
    "humidity":    (10,  99),
    "ph":          (4.0, 9.5),
    "rainfall":    (20,  300),
}

def _auto_feedback(label: str, confidence: float, inputs: dict) -> dict:
    """
    Generate automatic feedback without any human input.

    Rules
    -----
    * confidence >= 80 %  → agreed, high-confidence note
    * 60 ≤ confidence < 80 %  → agreed, moderate-confidence note
    * confidence < 60 %  → disagreed, low-confidence warning

    Any input that falls outside the agronomic gross-range table adds a
    flag to the comment so the feedback ledger surfaces suspicious sessions.
    """
    out_of_range = []
    for feat, (lo, hi) in _AGRO_RANGES.items():
        val = inputs.get(feat)
        if val is not None:
            try:
                v = float(val)
                if v < lo or v > hi:
                    out_of_range.append(f"{feat}={v:.2f} (expected {lo}–{hi})")
            except (TypeError, ValueError):
                pass

    range_note = (" | Unusual inputs: " + ", ".join(out_of_range)) if out_of_range else ""

    if confidence >= 80.0:
        agreed  = True
        comment = f"Auto: High-confidence prediction ({confidence:.1f}%){range_note}"
    elif confidence >= 60.0:
        agreed  = True
        comment = f"Auto: Moderate confidence ({confidence:.1f}%) — inputs are broadly consistent with {label}{range_note}"
    else:
        agreed  = False
        comment = f"Auto: Low confidence ({confidence:.1f}%) — inputs may be borderline or inconsistent for {label}{range_note}"

    return {"agreed": agreed, "comment": comment}


def _write_feedback_rows(rows: list[dict], ts_prefix: str = "") -> None:
    """
    Write a list of auto-feedback dicts to FEEDBACK_FP.

    Each dict must have keys:
      timestamp, N, P, K, temperature, humidity, ph, rainfall,
      predicted, confidence, agreed, comment
    (user_label is always empty, auto is always True)
    """
    new_fb = not os.path.exists(FEEDBACK_FP)
    with open(FEEDBACK_FP, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_fb:
            w.writerow(["timestamp", "N", "P", "K", "temperature",
                        "humidity", "ph", "rainfall",
                        "predicted", "confidence",
                        "agreed", "user_label", "comment", "auto"])
        for r in rows:
            w.writerow([r["timestamp"],
                        r.get("N"), r.get("P"), r.get("K"),
                        r.get("temperature"), r.get("humidity"),
                        r.get("ph"), r.get("rainfall"),
                        r["predicted"], r["confidence"],
                        r["agreed"], "",
                        r["comment"], True])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    ready = mu.is_ready()
    classes, features, stats = [], [], {}
    if ready:
        try:
            classes  = mu.get_class_list()
            features = mu.get_feature_list()
            stats    = mu.get_scaler_stats()
        except Exception as e:
            print("[init] artifacts present but failed to load:", e)
            ready = False

    n_feedback = 0
    n_history  = 0
    if os.path.exists(FEEDBACK_FP):
        try: n_feedback = sum(1 for _ in open(FEEDBACK_FP)) - 1
        except Exception: n_feedback = 0
    if os.path.exists(HISTORY_FP):
        try: n_history = sum(1 for _ in open(HISTORY_FP)) - 1
        except Exception: n_history = 0

    return render_template("index.html",
                           ready=ready,
                           classes=classes,
                           features=features,
                           stats=stats,
                           n_feedback=max(n_feedback, 0),
                           n_history=max(n_history, 0))


@app.route("/predict")
def predict_page():
    if not mu.is_ready():
        return redirect(url_for("index"))
    classes  = mu.get_class_list()
    features = mu.get_feature_list()
    stats    = mu.get_scaler_stats()
    return render_template("predict.html",
                           classes=classes,
                           features=features,
                           stats=stats)


@app.route("/predict/run", methods=["POST"])
def predict_run():
    """Internal route — called by JS for live updates. Returns JSON."""
    try:
        data = request.get_json(force=True)
        out = mu.predict_single(
            N=float(data.get("N",            90)),
            P=float(data.get("P",            42)),
            K=float(data.get("K",            43)),
            temperature=float(data.get("temperature", 22.0)),
            humidity=float(data.get("humidity",       80.0)),
            ph=float(data.get("ph",                    6.5)),
            rainfall=float(data.get("rainfall",       200.0)),
            n_runs=int(data.get("n_runs",               6)),
        )
        out["dossier"] = dossier_for(out["label"])

        ts = datetime.now().isoformat(timespec='seconds')

        # Append to history (lightweight log)
        new_hist = not os.path.exists(HISTORY_FP)
        with open(HISTORY_FP, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_hist:
                w.writerow(["timestamp", "N", "P", "K", "temperature",
                            "humidity", "ph", "rainfall",
                            "predicted", "confidence"])
            w.writerow([ts,
                        data.get("N"), data.get("P"), data.get("K"),
                        data.get("temperature"), data.get("humidity"),
                        data.get("ph"), data.get("rainfall"),
                        out["label"], round(out["confidence"], 2)])

        # Auto-feedback: evaluate every prediction immediately
        fb = _auto_feedback(out["label"], out["confidence"], data)
        _write_feedback_rows([{
            "timestamp":   ts,
            "N":           data.get("N"),  "P": data.get("P"),  "K": data.get("K"),
            "temperature": data.get("temperature"), "humidity": data.get("humidity"),
            "ph":          data.get("ph"), "rainfall": data.get("rainfall"),
            "predicted":   out["label"],
            "confidence":  round(out["confidence"], 2),
            **fb,
        }])

        out["auto_feedback"] = fb          # expose to JS so UI can reflect it
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if not mu.is_ready():
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("upload.html",
                               classes=mu.get_class_list(),
                               features=mu.get_feature_list())

    # POST — file upload, returns JSON for AJAX
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "No file selected"}), 400
    if not f.filename.lower().endswith((".csv", ".tsv")):
        return jsonify({"ok": False, "error": "Please upload a .csv file"}), 400

    save_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Could not read CSV: {e}"}), 400

    # Quick preview of preprocessing
    feats = mu.get_feature_list()
    missing = [c for c in feats if c not in df.columns]
    if missing:
        return jsonify({"ok": False,
                        "error": f"Missing columns: {missing}. "
                                 f"Required: {feats}"}), 400

    preview = df.head(10).to_dict(orient="records")
    desc    = df[feats].describe().to_dict()

    try:
        result = mu.predict_dataframe(df)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500

    # --- Auto-feedback for every uploaded row ----------------------------
    # Wipe previous feedback & history so the ledger reflects only this upload
    for _fp in (FEEDBACK_FP, HISTORY_FP):
        if os.path.exists(_fp):
            os.remove(_fp)

    feats   = mu.get_feature_list()
    batch_ts = datetime.now().isoformat(timespec='seconds')
    fb_rows  = []
    for row in result["rows"]:
        inputs = {feat: row.get(feat) for feat in feats}
        fb = _auto_feedback(row["predicted"], row["confidence"], inputs)
        fb_rows.append({
            "timestamp":   batch_ts,
            **inputs,
            "predicted":   row["predicted"],
            "confidence":  round(row["confidence"], 2),
            **fb,
        })
    _write_feedback_rows(fb_rows)

    # Feedback summary back to the UI
    n_agree    = sum(1 for r in fb_rows if r["agreed"])
    n_disagree = len(fb_rows) - n_agree
    result["auto_feedback_summary"] = {
        "total":     len(fb_rows),
        "agreed":    n_agree,
        "disagreed": n_disagree,
    }
    # ---------------------------------------------------------------------

    # Save predictions CSV alongside upload
    out_df = pd.DataFrame(result["rows"])
    out_path = os.path.join(UPLOAD_DIR,
                            os.path.splitext(f.filename)[0] + "_predictions.csv")
    out_df.to_csv(out_path, index=False)
    result["download"] = url_for("download_predictions",
                                 filename=os.path.basename(out_path))
    result["preview"]  = preview
    result["describe"] = desc

    return jsonify({"ok": True, "result": result})


@app.route("/download/<path:filename>")
def download_predictions(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, as_attachment=True, download_name=safe_name)


@app.route("/sample-csv")
def sample_csv():
    """A tiny example CSV in the model's expected schema (no API needed)."""
    rows = [
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"],
        [90, 42, 43, 20.87, 82.00, 6.50, 202.93, "rice"],
        [71, 54, 16, 22.61, 63.69, 5.75,  87.76, "maize"],
        [40, 72, 77, 17.02, 16.99, 7.49,  88.55, "chickpea"],
        [20, 30, 20, 25.00, 90.00, 6.50, 210.00, "rice"],
        [80, 45, 45, 27.00, 65.00, 6.80,  95.00, "maize"],
        [100, 20, 20, 30.00, 50.00, 7.00, 60.00, "cotton"],
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerows(rows)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name="sample_crop_input.csv")


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------
@app.route("/feedback", methods=["GET"])
def feedback_page():
    history_rows = []
    feedback_rows = []
    if os.path.exists(HISTORY_FP):
        try:
            history_rows = pd.read_csv(HISTORY_FP).tail(20).to_dict(orient="records")
            history_rows = list(reversed(history_rows))
        except Exception: pass
    if os.path.exists(FEEDBACK_FP):
        try:
            feedback_rows = pd.read_csv(FEEDBACK_FP).tail(40).to_dict(orient="records")
            feedback_rows = list(reversed(feedback_rows))
        except Exception: pass

    classes = mu.get_class_list() if mu.is_ready() else []

    # Aggregate metrics
    total = len(feedback_rows)
    agree = sum(1 for r in feedback_rows if str(r.get("agreed")).lower() == "true")
    disagree = total - agree
    agree_pct = (agree / total * 100) if total else 0.0
    n_auto   = sum(1 for r in feedback_rows if str(r.get("auto", "")).lower() == "true")
    n_manual = total - n_auto

    return render_template("feedback.html",
                           history=history_rows,
                           feedback=feedback_rows,
                           classes=classes,
                           total=total, agree=agree,
                           disagree=disagree, agree_pct=agree_pct,
                           n_auto=n_auto, n_manual=n_manual)


@app.route("/feedback/submit", methods=["POST"])
def feedback_submit():
    data = request.get_json(force=True)
    new = not os.path.exists(FEEDBACK_FP)
    with open(FEEDBACK_FP, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "N", "P", "K", "temperature",
                        "humidity", "ph", "rainfall",
                        "predicted", "confidence",
                        "agreed", "user_label", "comment", "auto"])
        w.writerow([datetime.now().isoformat(timespec='seconds'),
                    data.get("N"), data.get("P"), data.get("K"),
                    data.get("temperature"), data.get("humidity"),
                    data.get("ph"), data.get("rainfall"),
                    data.get("predicted"), data.get("confidence"),
                    bool(data.get("agreed", False)),
                    data.get("user_label", ""),
                    data.get("comment", ""), False])    # auto=False → manual
    return jsonify({"ok": True})


@app.route("/feedback/clear", methods=["POST"])
def feedback_clear():
    if os.path.exists(FEEDBACK_FP):
        os.remove(FEEDBACK_FP)
    return jsonify({"ok": True})


@app.route("/history/clear", methods=["POST"])
def history_clear():
    if os.path.exists(HISTORY_FP):
        os.remove(HISTORY_FP)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False,
                    "error": "File too large (max 16 MB)."}), 413


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("\n  Botanical Atlas — SNN Crop Recommendation Dashboard")
    print(f"  Model artifacts ready : {mu.is_ready()}")
    print(f"  Local  → http://127.0.0.1:5000")
    print(f"  Mobile → http://{local_ip}:5000  (same Wi-Fi network)\n")
    app.run(debug=True, use_reloader=False)