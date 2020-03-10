import json
import keras
import numpy as np
import scipy.io as sio
import scipy.stats as sst
import sys
import os
import load
import network
import util
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from mylimefortime import LimeTimeSeriesExplanation

STEP = 256

def loadmodel(mpath):
    model = keras.models.load_model(mpath)
    model.save_weights("model.hdf5")


def mypredict(ecg):

    preproc = util.load(".")
    x = preproc.process_x([ecg])
    params = json.load(open("config.json"))
    params.update({
        "compile": False,
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)
    model.load_weights('model.hdf5')

    probs = model.predict(x)
    classprobs = np.amax(probs, axis=1)
    prediction = sst.mode(np.argmax(probs, axis=2).squeeze())[0][0]

    predictionclass = preproc.int_to_class[prediction]
    print prediction, predictionclass
    return classprobs

def mylimepredict(ecg):

    preproc = util.load(".")
    x = preproc.process_x(ecg)
    params = json.load(open("config.json"))
    params.update({
        "compile": False,
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)
    model.load_weights('model.hdf5')

    probs = model.predict(x)
    classprobs = np.amax(probs, axis=1)
    prediction = sst.mode(np.argmax(probs, axis=2).squeeze())[0][0]

 #   predictionclass = preproc.int_to_class[prediction]
 #   print prediction, predictionclass
    return classprobs

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)
#    print ecg

    plt.plot(ecg[:len(ecg)])
    plt.show(block=False)
    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

def myexplain(ecg):

    explainer2 = LimeTimeSeriesExplanation(class_names=['0', '1', '2', '3'], feature_selection='auto')
    exp2 = explainer2.explain_instance(ecg, mylimepredict, num_features=num_features_ecg, num_samples=500,
                                       num_slices=num_slices_ecg,
                                       replacement_method='noise')
    exp2.as_list()

    values_per_slice_ecg = math.ceil(len(ecg) / num_slices_ecg)
    plt.plot(ecg, 'b', label='Explained instance (class 4)')
    print exp2.as_list()
    f = map(lambda x: x[0], exp2.as_list())
    w = map(lambda x: x[1], exp2.as_list())
    print w
    max_weight = abs(max(w, key=abs))
    print max_weight

    for i in range(num_features_ecg):
        feature, weight = exp2.as_list()[i]
        if(abs(weight) >= (max_weight-0.02)):
            start = feature * values_per_slice_ecg
            end = start + values_per_slice_ecg
            plt.axvspan(start, end, ec='red', fc= 'none')
#        plt.axvspan(start, end, color='red', alpha=abs(weight * 10))
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':

    num_features_ecg = 5
    num_slices_ecg = 15
    loadmodel(sys.argv[2])
    ecg = load_ecg(sys.argv[1] + ".mat")
    prediction = mypredict(ecg)
    print prediction
    myexplain(ecg)
    plt.show()