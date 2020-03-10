import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc
import numpy as np
from lime import explanation
from lime import lime_base
import math
from limefortime import LimeTimeSeriesExplanation


idx_ecg = 7
num_features_ecg = 5
num_slices_ecg = 17
series_ecg = ecg_test_x.iloc[idx_ecg, :]

explainer2 = LimeTimeSeriesExplanation(class_names=['0', '1'], feature_selection='auto')
exp2 = explainer2.explain_instance(series_ecg, knn2.predict_proba, num_features=num_features_ecg, num_samples=500, num_slices=num_slices_ecg,
                                 replacement_method='noise', training_set=ecg_train_x)
exp2.as_list()

values_per_slice_ecg = math.ceil(len(series_ecg) / num_slices_ecg)
plt.plot(series_ecg, 'b', label='Explained instance (class 4)')
plt.plot(ecg_test_x.iloc[ecg_test_y[ecg_test_y == 1].index, :].mean(), color='green',
        label='Mean of class 1')
plt.plot(ecg_test_x.iloc[ecg_test_y[ecg_test_y == 2].index, :].mean(), color='red',
        label='Mean of class 2')
plt.plot(ecg_test_x.iloc[ecg_test_y[ecg_test_y == 3].index, :].mean(), color='black',
        label='Mean of class 3')
for i in range(num_features_ecg):
    feature, weight = exp2.as_list()[i]
    start = feature * values_per_slice_ecg
    end = start + values_per_slice_ecg
    plt.axvspan(start , end, color='red', alpha=abs(weight*10))
plt.legend(loc='lower left')
plt.show()