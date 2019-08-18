import numpy as np
from flask import Flask, request, abort, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        try:
            model = joblib.load('model.pkl')
            data = request.get_json()
            if data is None:
                abort(400)
            
            X = np.array(data['payload']).reshape(1,-1)
            # numpy array is not JSON serializable, casting to list
            pred = model.predict(X).tolist()
            
            return jsonify({'prediction': pred})

        except (ValueError, TypeError) as e:
            return jsonify('Error with - {}'.format(e))

if __name__ == '__main__':
    app.run()