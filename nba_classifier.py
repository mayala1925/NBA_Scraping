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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from datetime import date
import os


today = date.today()
# Reading in the data.
data = pd.read_csv('nba_scrapping_data/schedule_ratings.csv')

# Drop rows with null values, these are the rows that were pulled that the games haven't been played yet...
classifier_data = data.dropna()

# Creating a target label for regression models.
classifier_data['a_wonby'] = classifier_data['away_pts'] - classifier_data['home_pts']

# Taking the rows that were dropped to make predictions.
pred_rows = data[data['date'] >= str(today)]
pred_features = pred_rows[pred_rows.columns[5:21]]

# Selecting the columns with the stats we want to use as features
features = classifier_data[classifier_data.columns[5:21]]
class_target = classifier_data[classifier_data.columns[-2]] # This target is the discrete variable of home or away team winning.
reg_target = classifier_data[classifier_data.columns[-1]] # This target variable is how much the away team won by.

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

X_train, X_test, y_train, y_test = train_test_split(features, class_target, test_size=0.2, random_state=0)

predictions = []
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    real_preds = pipe.predict(pred_features)
    predictions.append(real_preds)


# Getting the predicted winner that the most classifiers agree on.
final_predictions = []
for i in range(len(predictions[0])):
    pred_list = []

    for pred in predictions:
        pred_list.append(pred[i])
    home_count = pred_list.count('home')
    away_count = pred_list.count('away')
    if home_count > away_count:
        final_predictions.append('Home')
    elif home_count == away_count:
        final_predictions.append('Unsure')
    else:
        final_predictions.append('Away')

print(f'The classifiers predict... {final_predictions}')

# Putting the final predictions into the game rows row today.
pred_rows = pred_rows.drop('winner',axis = 1)
pred_rows['predicted_winner'] = final_predictions

site_table = pred_rows[['away','home','predicted_winner']]

for i, model in enumerate(pipelines):
    m_score = model.score(X_test,y_test)
    print(f'{pipe_dict[i]} Test Accuracy: {m_score}')


PATH_PARENT = os.path.dirname(os.getcwd())
path = os.path.join(PATH_PARENT,'personal-site/static/predictions')

site_table.to_csv(f'{path}/predictions.csv',index = False)


#%%
# Commented out lines are for kfold train/test splits, should implement later.
kf = KFold(n_splits=N_SPLITS)

# print(pipeline_lda.predict(X_test))
# print(y_test)

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

#%%
# Trying Gradient Boosting Classifier
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

gbm = GradientBoostingClassifier(random_state=1, n_estimators=120, learning_rate= 0.1, max_depth=3)
cv = StratifiedKFold(n_splits=5,random_state=1, shuffle=True)

n_scores = cross_val_score(gbm, scaled_features, y=class_target, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')



print(f'Gradient Boosting Accuracy: {np.mean(n_scores)}, {np.std(n_scores)}')

#%%
# Creating multiple regression models.

print('NOW FOR THE REGRESSION MODELS...')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Scaling data first

scaled_features = scaler.fit_transform(features)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(features, reg_target, test_size=0.2, random_state=0)

lm_model = LinearRegression()

lm_model.fit(X_train_reg,y_train_reg)

# print(f'Multiple Linear Regression Score: {lm_model.score(X_test_reg,y_test_reg)}')

# Creating pipelines
pipeline_lm = Pipeline([('scalar1', StandardScaler()),
                        ('lm_classifier', LinearRegression())])

pipeline_r = Pipeline([('scalar2', StandardScaler()),
                        ('r_classifier', Ridge())])

pipeline_dtr = Pipeline([('scalar3', StandardScaler()),
                        ('dtr_classifier', DecisionTreeRegressor(random_state=0))])

pipeline_svr = Pipeline([('scalar3', StandardScaler()),
                        ('svr_classifier', SVR())])

pipelines_reg = [pipeline_lm, pipeline_r, pipeline_dtr, pipeline_svr]

# best_accuracy = 0.0
# best_classifier = 0
# best_pipeline = ""

pipe_dict_reg = {0: 'Linear Regression',
             1: 'Ridge',
             2: 'Decision Tree Regressor',
             3: 'Support Vector Regressor'
                 }

for pipeline in pipelines_reg:
    pipeline.fit(X_train_reg, y_train_reg)

for i, model in enumerate(pipelines_reg):
    reg_pred = model.predict(X_test_reg)
    mse_score = mean_squared_error(y_test_reg,reg_pred)
    r_score = r2_score(y_test_reg,reg_pred)
    print(f'{pipe_dict_reg[i]}: Mean Squared Error: {mse_score},R2:{r_score}')