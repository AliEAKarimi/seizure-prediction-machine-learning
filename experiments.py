import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
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

sampling_freq = 173.6 #based on info from website

b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)


x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])

x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal,x_seizure))
y = np.concatenate((np.zeros((400,1)),np.ones((100,1))))

def plot(x, title):
    plt.plot(x)
    plt.title(title)
    plt.show()

def evaluation(test, pred):
    print('Accuracy: ', accuracy_score(test, pred))
    print('Recall: ', recall_score(test, pred))
    print('Precision: ', precision_score(test, pred))

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

mode = []
for row in x:
    mode.append(st.mode(row, keepdims=True)[0][0])
mode = np.array(mode)
plot(mode, 'mode')

# Time domain features: mobility, complexity, average absolute signal slope, peak-to-peak
def hjorth_params(x, axis=-1):
    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    return mob, com

mobility, complexity = hjorth_params(x)
plot(mobility, 'mobility')
plot(complexity, 'complexity')


peak_to_peak = np.ptp(x, axis=1)
plot(peak_to_peak, 'peak_to_peak')

average_absolute_signal_slope = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
plot(average_absolute_signal_slope, 'average_absolute_signal_slope')

# Frequency domain features: Delta, Theta, Alpha, Beta, Gamma
def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.
    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If window_sec=None, window_sec = (1 / min(band)) * 2
    relative : bool
        If relative is True, return the relative power (= divided by the total power of the signal).
    Returns
    -------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps

    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

delta = []
theta = []
alpha = []
beta = []
gamma = []

for row in x:
    delta.append(bandpower(row, 128, [0.5, 4], 4))
    theta.append(bandpower(row, 128, [4, 8], 4))
    alpha.append(bandpower(row, 128, [8, 13], 4))
    beta.append(bandpower(row, 128, [13, 30], 4))
    gamma.append(bandpower(row, 128, [30, 50], 4))
delta = np.array(delta)
theta = np.array(theta)
alpha = np.array(alpha)
beta = np.array(beta)
gamma = np.array(gamma)
plot(delta, 'delta')
plot(theta, 'theta')
plot(alpha, 'alpha')
plot(beta, 'beta')
plot(gamma, 'gamma')

x_visualized = np.array([mean, std, max, min, median, var, skewness, kurtosis, mobility, complexity, peak_to_peak, average_absolute_signal_slope, delta, theta, alpha, beta, gamma])
x_visualized = x_visualized.T
#print(visualized_x.shape)
plot(x_visualized, 'X Visualized')

####################### Classification #######################
# using train_test_split to split data into train and test
xx_train, xx_test, yy_train, yy_test = train_test_split(x,y,random_state=seed,test_size=0.2)
print(xx_train.shape)
print(xx_test.shape)

# using train_test_split to split visualized data into train and test
x_train, x_test, y_train, y_test = train_test_split(x_visualized,y,random_state=seed,test_size=0.2)
print(x_train.shape)
print(x_test.shape)

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

kf = KFold(n_splits=5, random_state=seed, shuffle=True)

def cross_validation(clf, x, y):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
    }
    scores = cross_validate(clf, x, y, cv=kf, scoring=scoring)
    print(scores)
    print('Accuracy: ', np.mean(scores['test_accuracy']))
    print('Recall: ', np.mean(scores['test_recall']))
    print('Precision: ', np.mean(scores['test_precision']))

# using SVM
# svm with linear kernel
linear_svm_clf = SVC(kernel='linear')
linear_svm_clf.fit(x_train, y_train)
svm_y_pred = linear_svm_clf.predict(x_test)
evaluation(y_test,svm_y_pred)
# svm with rbf kernel
rbf_svm_clf = SVC(kernel='rbf')
rbf_svm_clf.fit(x_train, y_train)
svm_y_pred = rbf_svm_clf.predict(x_test)
evaluation(y_test,svm_y_pred)
# svm with poly kernel
poly_svm_clf = SVC(kernel='poly')
poly_svm_clf.fit(x_train, y_train)
svm_y_pred = poly_svm_clf.predict(x_test)
evaluation(y_test,svm_y_pred)
# svm with sigmoid kernel
sigmoid_svm_clf = SVC(kernel='sigmoid')
sigmoid_svm_clf.fit(x_train, y_train)
svm_y_pred = sigmoid_svm_clf.predict(x_test)
evaluation(y_test,svm_y_pred)

# using Random Forest
random_forest_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=seed)
random_forest_clf.fit(x_train, y_train)
random_forest_y_pred = random_forest_clf.predict(x_test)
evaluation(y_test,random_forest_y_pred)

# using KNN
knn_clf = KNeighborsClassifier(n_neighbors=2)
knn_clf.fit(x_train, y_train)
knn_y_pred = knn_clf.predict(x_test)
evaluation(y_test,knn_y_pred)

# Drawing ROC curve
y_score = linear_svm_clf.predict_proba(x_test)[::,1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,  y_score)
auc = roc_auc_score(y_test, y_score)
plt.plot(false_positive_rate,true_positive_rate,label="auc="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Drawing confusion matrix for svm_clf
cm = confusion_matrix(y_test, svm_y_pred, normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()