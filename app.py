import os
import numpy as np
import json
from flask import Flask, request, jsonify, make_response
import tensorflow as tf
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, static_folder='../dist/static')
CORS(app)
MODEL_PATH = 'models/detectx-mobilenet-40.hdf5'
#MODEL_JSON_PATH = 'models/model.json'

#loaded_json_model = None
#with open(MODEL_JSON_PATH) as json_model:
#	loaded_model_json = json_model.read()
#model = model_from_json(loaded_model_json)
#print(model.summary())
model = None

def load_detect_model():
	global model
	MODEL_PATH = 'models/detectx-mobilenet-40.hdf5'
	model = load_model(MODEL_PATH)
	
def predict(img_file):
	global model
	img = image.load_img(img_file, target_size=(150,150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	# print("X", x.shape)
	x = preprocess_input(x, mode='keras')
	#graph = tf.get_default_graph()
	#prediction = None
	#with graph.as_default():
	# model = load_model(MODEL_PATH)
	prediction = model.predict(x)
		# print("Prediction", prediction)
	pred_class = np.argmax(prediction[0])
	label = "Not Dangerous" if pred_class==1 else "Dangerous"
	detectDanger = False if pred_class==1 else True
	score = prediction[0][pred_class] * 100
	return label, score, detectDanger

@app.route("/classify", methods=["GET","POST"])
def classify():
	if request.method == "OPTIONS": # CORS preflight
	    return _build_cors_prelight_response()
	elif request.method == "POST":
		# print("Input", request.files)
		img = request.files['image']
		# basepath = os.path.dirname(__file__)
		# img_path = os.path.join(basepath, 'uploads', secure_filename(img.filename))
		# img.save(img_path)
		label, score, detectDanger = predict(img)
		result = [
        	{
                'label': label,
                'score': score,
				'detectDanger': detectDanger
        	}
		]
		print('Response generated')
		return jsonify(result)
	return None

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

if __name__ == "__main__":
	load_detect_model()
	app.run()
