from flask import Flask, request, jsonify
import pandas as pd
# import request
import sys
from sklearn.externals import joblib
import traceback

from tensorflow.keras.models import load_model
from modules.package import convert_to_vectors, padding


app = Flask(__name__)


@app.route('/', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    if best_model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query["review"]
            text_converted = convert_to_vectors(query)
            text_converted = padding(text_converted)
            predict = list(best_model.predict(text_converted))

            return jsonify({'prediction': predict})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:

        return ("Model not found")

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    best_model = load_model("model/")
    print('Model loaded')
    app.run(port=port, debug=True)