# -*- coding: utf-8 -*-

import numpy as np

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
import tensorflow as tf
from keras import backend as K
import pdb
import json

from vis_utils import text_util
from vis_utils.hanVis import HNATT

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


SAVED_MODEL_DIR = './vis_utils/saved_models'
SAVED_MODEL_FILENAME = 'model'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

K.clear_session()
hh = HNATT()
hh.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
graph = tf.get_default_graph()

@app.route('/')
def hhello_world():
    return render_template('index.html')

@app.route('/activations')
def activations():
    """
    Receive a text and return hhNATT activation map
    """
    if request.method == 'GET':
        text = request.args.get('text', '')
        if len(text.strip()) == 0:
            return Response(status=400)

        encoded_text = text_util.encode_input(text)
        s_text = text_util.splitAll(text)

        global graph
        with graph.as_default():
            activation_maps = hh.activation_maps(text, websafe=True)
            preds = hh.predict(encoded_text)[0]
            prediction = np.argmax(preds).astype(float)
            data = {
                'activations': activation_maps,
                'normalizedText': s_text,
                'prediction': prediction,
                'binary': preds.shape[0] == 2
            }
            #pdb.set_trace()
            return jsonify(data)
#            res = json.dumps(data, ensure_ascii=False)
#            return res
    else:
        return Response(status=501)
