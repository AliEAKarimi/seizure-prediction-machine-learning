import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from scipy import stats as st

import random
import os
seed = 57

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x_normal = np.concatenate((x[:300], x[400:]), axis=0)
x_seizure = x[300:400]
# print(x_normal.shape)
# print(x_seizure.shape)
sampling_freq = 173.6 #based on info from website

b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)


x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])
# print(x_normal.shape)
# print(x_seizure.shape)


x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal,x_seizure))
y = np.concatenate((np.zeros((400,1)),np.ones((100,1))))
# print(x.shape)
# print(y.shape)

def plot(x, title):
    plt.plot(x)
    # plt.title(title)
    # plt.show()

####################### Feature Extraction #######################
# Statistical features: mean, std, max, min, median, variance, skewness, kurtosis, mode
mean = np.mean(x, axis=1)
plot(mean, 'mean')

std = np.std(x, axis=1)
plot(std, 'std')

max = np.max(x, axis=1)
plot(max, 'max')

min = np.min(x, axis=1)
plot(min, 'min')

median = np.median(x, axis=1)
plot(median, 'median')

var = np.var(x, axis=1)
plot(var, 'var')

skewness = st.skew(x, axis=1)
plot(skewness, 'skewness')

kurtosis = st.kurtosis(x, axis=1)
plot(kurtosis, 'kurtosis')

# mode = []
# for row in x:
#     mode.append(st.mode(row)[0][0])
# mode = np.array(mode)
# print(mode.shape)
# print(mode)
# plt.plot(mode)
# plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=seed,test_size=0.2)

kf  = KFold(n_splits=5,random_state=seed,shuffle=True)
kf.get_n_splits(x)
# print(kf)
for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
# print(x_test.shape)

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_test,y_pred))





