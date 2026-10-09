from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from pathlib import Path


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "model.pkl পাওয়া যায়নি। "
        "প্রথমে terminal-এ 'python train_model.py' চালান।"
    )


model = joblib.load(MODEL_PATH)


# ==========================================
# MODEL FEATURES
# ==========================================

FEATURES = [
    "bedrooms",
    "bathrooms",
    "floors",
    "year_built",
    "area_sqft"
]


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------
        # Receive JSON data
        # ----------------------------------

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "success": False,
                "error": "কোনো data পাওয়া যায়নি।"
            }), 400


        # ----------------------------------
        # Check required fields
        # ----------------------------------

        missing_fields = []

        for field in FEATURES:

            if field not in data:

                missing_fields.append(field)

            elif data[field] in ("", None):

                missing_fields.append(field)


        if missing_fields:

            return jsonify({
                "success": False,
                "error":
                    "এই fieldগুলো পূরণ করুন: "
                    + ", ".join(missing_fields)
            }), 400


        # ----------------------------------
        # Convert values to float
        # ----------------------------------

        bedrooms = float(data["bedrooms"])

        bathrooms = float(data["bathrooms"])

        floors = float(data["floors"])

        year_built = float(data["year_built"])

        area_sqft = float(data["area_sqft"])


        # ==================================
        # INPUT VALIDATION
        # ==================================

        if not 1 <= bedrooms <= 20:

            raise ValueError(
                "Bedrooms 1 থেকে 20-এর মধ্যে হতে হবে।"
            )


        if not 1 <= bathrooms <= 20:

            raise ValueError(
                "Bathrooms 1 থেকে 20-এর মধ্যে হতে হবে।"
            )


        if not 1 <= floors <= 10:

            raise ValueError(
                "Floors 1 থেকে 10-এর মধ্যে হতে হবে।"
            )


        if not 1800 <= year_built <= 2026:

            raise ValueError(
                "Year Built 1800 থেকে 2026-এর মধ্যে হতে হবে।"
            )


        if not 200 <= area_sqft <= 20000:

            raise ValueError(
                "Area 200 থেকে 20,000 sqft-এর মধ্যে হতে হবে।"
            )


        # ==================================
        # PREPARE INPUT FOR MODEL
        # ==================================

        input_data = np.array([[
            bedrooms,
            bathrooms,
            floors,
            year_built,
            area_sqft
        ]], dtype=np.float64)


        # ==================================
        # MACHINE LEARNING PREDICTION
        # ==================================

        prediction = model.predict(input_data)


        predicted_price = float(prediction[0])


        # ==================================
        # SAFETY CHECK
        # ==================================

        if predicted_price < 0:

            predicted_price = 0


        # ==================================
        # FORMAT PRICE
        # ==================================

        formatted_price = (
            f"৳{predicted_price:,.0f}"
        )


        # ==================================
        # RESPONSE
        # ==================================

        return jsonify({

            "success": True,

            "price": round(
                predicted_price,
                2
            ),

            "formatted_price":
                formatted_price

        })


    # ======================================
    # INPUT ERROR
    # ======================================

    except ValueError as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


    # ======================================
    # OTHER ERROR
    # ======================================

    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

        return jsonify({

            "success": False,

            "error":
                "Prediction করার সময় একটি সমস্যা হয়েছে।"

        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )