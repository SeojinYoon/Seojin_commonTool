
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
    # JS에서 fetch("http://localhost:5000/click")로 호출하고 있으니 포트 5000으로 실행
    app.run(host="0.0.0.0", port=5000, debug=True)


