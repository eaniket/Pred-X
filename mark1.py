import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
#import os

df = pd.read_csv("heart.csv")

col_name = set(df.columns)
col_name.remove('target')
X = df[col_name]
# X = df.columns
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#### Commented due to pickling
# clf = linear_model.SGDClassifier(penalty='l2')
# clf.fit(X_train, y_train)

#pickle.dump(clf,open('classifier.pkl','wb'), protocol=4)

clf = pickle.load(open('classifier.pkl','rb'))

y_pred = clf.predict(X_test)
print(np.mean(y_pred == y_test))