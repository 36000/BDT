import numpy as np
import matplotlib.pyplot as plt

tpr = np.genfromtxt('tpr.csv', delimiter=',')
fpr = np.genfromtxt('fpr.csv', delimiter=',')

AUC = np.trapz(tpr, x=fpr)
st_AUCb = "AUCb = " + str('%.2f' % round(AUC, 2))

pltRoc = plt.plot(fpr, tpr, linewidth=2, drawstyle='steps-post', color='blue')
pltDiag = plt.plot([0,1],[0,1], 'r--')

tpr = np.genfromtxt('tprc.csv', delimiter=',')
fpr = np.genfromtxt('fprc.csv', delimiter=',')

AUC = np.trapz(tpr, x=fpr)
st_AUCc = "AUCc = " + str('%.2f' % round(AUC, 2))

pltRoc = plt.plot(fpr, tpr, linewidth=2, drawstyle='steps-post', color='green')
pltDiag = plt.plot([0,1],[0,1], 'r--')
plt.text(0.7,0.2, st_AUCb, fontsize=17, weight=550)
plt.text(0.7,0.1, st_AUCc, fontsize=17, weight=550)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('false positive rate (fpr)', fontsize=15)
plt.ylabel('true positive rate (tpr)', fontsize=15)
plt.title('BDT Performance - Expert Level Variables Only', fontsize=19)
plt.savefig('test_ML_ROC_bdt.png')