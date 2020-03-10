import collections
import json
import keras
import numpy as np
import os
import sys
#sys.path.append("../../../ecg")
import scipy.stats as sst

import util
import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn.metrics as skm
import scikitplot as skplt

model_path = "bestmodel.hdf5"
data_path = "dev.json"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
model = keras.models.load_model(model_path)

data_path = "train.json"
with open("train.json", 'r') as fid:
    train_labels = [json.loads(l)['labels'] for l in fid]
counts = collections.Counter(preproc.class_to_int[l[0]] for l in train_labels)
counts = sorted(counts.most_common(), key=lambda x: x[0])
counts = zip(*counts)[1]
smooth = 500
counts = np.array(counts)[None, None, :]
total = np.sum(counts) + counts.shape[1]
prior = (counts + smooth) / float(total)
probs = []
labels = []
for x, y  in zip(*data):
    x, y = preproc.process([x], [y])
    probs.append(model.predict(x))
    labels.append(y)
preds = []
ground_truth = []
for p, g in zip(probs, labels):
    preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
    ground_truth.append(sst.mode(np.argmax(g, axis=2).squeeze())[0][0])

print "preds"
print preds
print "ground truth"
print ground_truth
#print "single value"
#print preds[2]

#n = np.random.binomial(len(preds), 0.07)
#preds = ground_truth
#preds[np.random.randint(0, len(preds), size=n)] += 2 % 4

#print preds

report = skm.classification_report(
            ground_truth, preds,
            target_names=preproc.classes,
            digits=3)
scores = skm.precision_recall_fscore_support(
            ground_truth,
            preds,
            average=None)
print(report)
print "CINC Average {:3f}".format(np.mean(scores[2][:3]))
print preproc.classes
ecglabels = ['A', 'N', 'O', '~']
skplt.metrics.plot_confusion_matrix(ground_truth, preds, title="Confusion Matrix XAI2ROI ECG model", normalize=True)
plt.show()