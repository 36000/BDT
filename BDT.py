import h5py
import numpy as np

#filepath = '/phys/groups/tev/scratch1/users/jk232/'
f = h5py.File('../'+'gjj_Variables.hdf5', 'r')

'''#for testing
f = h5py.File(filepath+'mygjj_Variables.hdf5', 'r')'''

high = f['high_input'][0:10000000]
y = f['y_input'][0:10000000]

#masking nan
print 'masking nan...'

#for high
mask = ~np.isnan(high).any(axis=2)
high_input = high[mask[:,0],...]
y_input = y[mask[:,0],...]

n_samples = high_input.shape[0]
tt_split = 0.8

shuff_index = np.arange(n_samples)
np.random.shuffle(shuff_index)

print 'reshaping...'

high_input = np.reshape(high_input, (n_samples,high_input.shape[1]*high_input.shape[2]))

yt_input = np.empty([n_samples])
for i in range(0,n_samples):
    for j in range(0, 3):
        if y_input[i, j] == 1:
            yt_input[i] = j
            break

yi_input = y_input
y_input = yt_input

print 'Train Test Split and shuffle...'

Train_Range = shuff_index[0:int(tt_split * n_samples)]
Test_Range = shuff_index[int(tt_split * n_samples) :]

X_train = high_input[Train_Range, :]
X_test = high_input[Test_Range, :]

y_train = y_input[Train_Range]
y_test = yi_input[Test_Range, :]

print 'training..'

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=1000)

import xgboost
clf = xgboost.XGBClassifier()
clf = clf.fit(X_train, y_train, verbose = True)
y_hat = clf.predict_proba(X_test)

print y_hat[0:10]
print y_test[0:10]

'''ROC Curve (Recall)
True Positive Rate (tpr) = TP / P = TP / (TP + FN)
False Positive Rate (fpr) = FP / N = FP / (FP + TN) = 1 - TN / (FP + TN)

for classification:
TPR = P(test positive | is bottom jet)
FPR = P(test negative | not bottom jet)

Finding TPR:
Set probability Threshold
collect all the true signal in y
for corresponding y_hat, see how many y_hats are above Threshold'''

tpr = []
fpr = []

for i in range(100):
    th = i/float(100)
    TP = np.sum((y_hat[:,2] >= th) * y_test[:,2])
    tpr.append( TP / float(np.sum(y_test[:,2])) )

    TN = np.sum((y_hat[:,2] < th) * (1-y_test[:,2]))
    fpr.append( 1 - TN / float(np.sum(y_test[:,0]+y_test[:,1])) )

tpr = np.concatenate([[0.0], tpr])
fpr = np.concatenate([[0.0], fpr])

tprc = []
fprc = []

for i in range(100):
    th = i/float(100)
    TP = np.sum((y_hat[:,1] >= th) * y_test[:,1])
    tprc.append( TP / float(np.sum(y_test[:,1])) )

    TN = np.sum((y_hat[:,1] < th) * (1-y_test[:,1]))
    fprc.append( 1 - TN / float(np.sum(y_test[:,0]+y_test[:,2])) )

tprc = np.concatenate([[0.0], tprc])
fprc = np.concatenate([[0.0], fprc])

np.savetxt("tpr.csv", np.sort(tpr), delimiter=',')
np.savetxt("fpr.csv", np.sort(fpr), delimiter=',')
np.savetxt("tprc.csv", np.sort(tprc), delimiter=',')
np.savetxt("fprc.csv", np.sort(fprc), delimiter=',')
