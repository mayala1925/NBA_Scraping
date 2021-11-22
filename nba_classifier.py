import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Reading in the data.
data = pd.read_csv('nba_scrapping_data/schedule_ratings.csv')

# Drop rows with null values, these are the rows that were pulled that the games haven't been played yet...
classifier_data = data.dropna()

# Taking the rows that were dropped to make predictions.
pred_rows = data[data['date'] >= '2021-11-22']
pred_features = pred_rows[pred_rows.columns[5:13]]

# Encoding labels to 1's and 0's
# label_codes = {'home':1, 'away':0}
#
# classifier_data['winner'] = classifier_data['winner'].map(label_codes)

# Selecting the columns with the stats we want to use as features
features = classifier_data[classifier_data.columns[5:13]]


N_SPLITS = 5


# Creating pipelines
pipeline_lr = Pipeline([('scalar1', StandardScaler()),
                        ('lr_classifier', LogisticRegression(random_state=0))])

pipeline_kn = Pipeline([('scalar2', StandardScaler()),
                        ('kn_classifier', KNeighborsClassifier())])

pipeline_dt = Pipeline([('scalar3', StandardScaler()),
                        ('dt_classifier', DecisionTreeClassifier(random_state=0))])

pipeline_lda = Pipeline([('scalar3', StandardScaler()),
                        ('lda_classifier', LinearDiscriminantAnalysis())])

pipeline_svc = Pipeline([('scalar3', StandardScaler()),
                        ('svc_classifier', SVC(random_state=0))])

pipelines = [pipeline_lr, pipeline_kn, pipeline_dt, pipeline_lda, pipeline_svc]

best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""

pipe_dict = {0: 'Logistic Regression',
             1: 'KNeighbors',
             2: 'Decision Tree',
             3: 'Linear Discriminant Analysis',
             4: 'Support Vector Machine'}

X_train, X_test, y_train, y_test = train_test_split(features, classifier_data['winner'], test_size=0.2, random_state=0)


kf = KFold(n_splits=N_SPLITS)
# acc_score = []

# for train_index, test_index in kf.split(features):
#     X_train, X_test = features.iloc[train_index,:], features.iloc[test_index,:]
#     y_train, y_test = classifier_data['winner'][train_index], classifier_data['winner'][test_index]
#
#     pipeline_lda.fit(X_train,y_train)
#     pred_values = pipeline_lda.predict(X_test)
#
#     acc = accuracy_score(pred_values, y_test)
#     acc_score.append(acc)
#
# avg_score = sum(acc_score) / N_SPLITS
#
# print(f'Accuracy of each fold: {acc_score}')
# print(f'Average accuracy score: {avg_score}')
# predictions_df = pd.DataFrame()



predictions = []
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    real_preds = pipe.predict(pred_features)
    predictions.append(real_preds)


for i, model in enumerate(pipelines):
    print(f'{pipe_dict[i]} Test Accuracy: {model.score(X_test,y_test)}')




# print(pipeline_lda.predict(X_test))
# print(y_test)