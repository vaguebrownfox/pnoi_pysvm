# import modules
import numpy as np
import pickle

# load svm model
with open('model.sav', 'rb') as m:
    lmodel = pickle.load(m)

# feature vector from mfcc coeffs (here using dummy array instead, just to check dimensions)
ft = np.zeros(72).reshape(1, 72)

# predict output
pred = lmodel.predict(ft)
print("asthma" if pred[0] == 0 else "no asthma")
