# xai2roi
XAI2ROI Using post-hoc explainable AI method to improve accuracy by focusing on regions of interest 

To run on ECG data, first download and train the models at https://github.com/awni/ecg/tree/master/ecg
*You will need to fix the data path in buildatasets.py

Gauge the accuracy of the model in its classifications
Run myexplainability.py which will call mylime and find the regions of importance based on the size parameters

Once done, run myroipool.py with the parameter set to the x,y bounding box around the roi found in the LIME code
Gauge the delta increase in accuracy

To run on Echo data, first download and train the models at https://github.com/douyang/EchoNetDynamic/
