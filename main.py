import pandas as pd

FinalDataFrame = pd.read_excel("FinaldataFrame1.xlsx")
from sklearn.preprocessing import LabelEncoder
for column in FinalDataFrame.columns:
    if FinalDataFrame[column].dtype == type(object):
        le = LabelEncoder()
        FinalDataFrame[column] = le.fit_transform(FinalDataFrame[column])
FinalDataFrame.dtypes

# cols = FinalDataFrame.select_dtypes(exclude=['float']).columns

# FinalDataFrame[cols] = FinalDataFrame[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
FinalDataFrame.dtypes


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

X = FinalDataFrame[
    ['Column1', 'Column2', 'Cosine Similarity', 'N-gramCol1', 'N-gramCol2', 'Synonyms_Col1', 'Synonyms_Col2',
     'TF-IDF Score', 'TaggedWordsCol1', 'TaggedWordsCol2']]
y = FinalDataFrame['IsParaphrased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
AdaBoost = AdaBoostClassifier(base_estimator=classifier, n_estimators=400, learning_rate=1)
AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
Ada = AdaBoost.fit(X_train, y_train)
prediction = Ada.predict(X_test)

# clf = classifier.fit(X_train, y_train)

# target_pred = clf.predict(X_test)

print(confusion_matrix(y_test, prediction))

accuracy_score = accuracy_score(y_test, prediction)
print(accuracy_score)

f1_score = f1_score(y_test, prediction, average='macro')
print(f1_score)

recall_score = recall_score(y_test, prediction, average='macro')
print(recall_score)

precision_score = precision_score(y_test, prediction, average='macro')
print(precision_score)

print('Average accuracy: %0.2f +/- (%0.1f) %%' % (accuracy_score.mean() * 100, accuracy_score.std() * 100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (precision_score.mean() * 100, precision_score.std() * 100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (recall_score.mean() * 100, recall_score.std() * 100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (f1_score.mean() * 100, f1_score.std() * 100))





import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = FinalDataFrame[['Column1', 'Column2', 'Cosine Similarity', 'N-gramCol1', 'N-gramCol2', 'Synonyms_Col1', 'Synonyms_Col2', 'TF-IDF Score', 'TaggedWordsCol1', 'TaggedWordsCol2']]
y = FinalDataFrame['IsParaphrased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(random_state=0)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1)

model_real = bdt.fit(X_train, y_train)

predict_real = model_real.predict(X_test)


print(confusion_matrix(y_test, predict_real))


accuracy_score = accuracy_score(y_test, predict_real)
print(accuracy_score)


f1_score = f1_score(y_test, predict_real, average='macro')
print(f1_score)


recall_score = recall_score(y_test, predict_real, average='macro')
print(recall_score)

precision_score = precision_score(y_test, predict_real, average='macro')
print(precision_score)


print('Average accuracy: %0.2f +/- (%0.1f) %%' % (accuracy_score.mean()*100, accuracy_score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (precision_score.mean()*100, precision_score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (recall_score.mean()*100, recall_score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (f1_score.mean()*100, f1_score.std()*100))

from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import numpy as np

Voting = VotingClassifier(estimators=[('knn', classifier), ('DT', clf)], voting='soft')

probas = [c.fit(X_train, y_train).predict_proba(X_test) for c in (classifier, clf, Voting)]
print(probas)

class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]
print(class1_1)
print(class2_1)

N = 3  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots()

# bars for classifier
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')

# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')

# plot annotations
plt.axvline(1.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['KNN', 'DecisionTree', 'VotingClassifier\n(average probabilities)'], rotation=40, ha='right')

plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.tight_layout()
plt.show()