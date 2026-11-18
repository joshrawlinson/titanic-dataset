import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


# create dataframe from dataset CSV w/ only relevant cols
df = pd.read_csv("Titanic-Dataset.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]

# drop rows with missing data for now
df = df.dropna()
df['Sex'] = df['Sex'].map({"male":1, "female":0})
df['Embarked'] = df["Embarked"].map({"S":0, "C":1, "Q":2})
print(df)

X_train, X_eval, y_train, y_eval = train_test_split(df[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']], df['Survived'])


clf = tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(X_train, y_train)

pred = clf.predict(X_eval)
print(pred)

print(X_train)