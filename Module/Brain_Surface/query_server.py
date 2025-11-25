
# Common Libraries
import sys
from pathlib import Path
from flask_cors import CORS
from flask import Flask, request, jsonify

# Custom Libraries
sys.path.append(str(Path(__file__).resolve().parent.parent))
from afni_extension import whereami

app = Flask(__name__)
CORS(app)

@app.route("/click", methods=["POST"])
def click_callback():
    data = request.get_json()  # JS에서 보낸 JSON 받기

    hemi = data.get("hemi")
    idx = data.get("idx")
    mni = data.get("mni")

    atlas_data = whereami(x = mni[0], y = mni[1], z = mni[2])
    return jsonify({
        "status": "ok",
        "message": "Click received",
        "hemi": hemi,
        "idx": idx,
        "mni": mni,
        "atlas_data" : atlas_data.to_dict(orient="records")
    })

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("⚠️ Port must be an integer. Using default port 5000.")

    app.run(host="0.0.0.0", port=5000, debug=True)

