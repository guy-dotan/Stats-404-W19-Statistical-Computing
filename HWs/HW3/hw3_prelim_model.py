'''
     File name: hw3_prelim_model.py
     Author: Guy Dotan
     Date: 02/11/2019
     Course: UCLA Stats 404
     Description: HW #3. Preliminary model to predict _____ from _____.
 '''

from collections import Counter
import inspect
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, \
                            mean_squared_error, roc_auc_score
import os

path = 'social-power-nba/'
df = pd.read_csv(path + "nba_2017_players_with_salary_wiki_twitter.csv")

#fullset = df.dropna(subset=['TWITTER_FAVORITE_COUNT'], inplace=True)


Counter(df['TWITTER_RETWEET_COUNT'])

fulldata = df[np.isfinite(df['TWITTER_RETWEET_COUNT'])]

# calculate a 5-number summary
# generate data sample
data = fulldata['TWITTER_RETWEET_COUNT']
# calculate quartiles
quartiles = np.percentile(data, [25, 50, 75])
# calculate min/max
data_min, data_max = data.min(), data.max()
# print 5-number summary
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)


notebook_dir = os.getcwd()
notebook_dir

#df = pd.DataFrame({'Age': [99, 53, 71, 84, 84],
#                   'Age_units': ['Y', 'Y', 'Y', 'Y', 'Y']})

bins = [0, 10, 50, 150, 3000]
names = ['<10', '10-49', '50-149', '300+']
d = dict(enumerate(names, 1))

nba = fulldata

nba['TWEET_CAT'] = np.vectorize(d.get)(np.digitize(fulldata['TWITTER_RETWEET_COUNT'], bins))

Counter(nba['TWEET_CAT'])

print(d)

nba.columns


df_train, df_valid = train_test_split(nba,
                                      test_size=0.25,
                                      random_state=2019,
                                      stratify=nba['TWEET_CAT'])


df_train['TWEET_CAT'].value_counts(sort=True)
df_valid['TWEET_CAT'].value_counts(sort=True)



y = df_train['TWEET_CAT']
X = df_train[['AGE','WINS_RPM','SALARY_MILLIONS','W', 'ORPM', 'DRPM','PIE']]

#X = df_train.drop(columns=['Rk', 'Unnamed: 0', 'PLAYER', 'POSITION', 'PAGEVIEWS',
#                           'TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT', 'TEAM', 'TWEET_CAT'])

X.shape
X.columns

X.isna().any()

#X[['3P%','FT%']] = X[['3P%','FT%']].fillna(0)

### --- Step 1: Specify different number of trees in forest, to determine
###             how many to use based on leveling-off of OOB error:
n_trees = [50, 100, 250, 500, 1000, 1500, 2500]


### --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)


for num in n_trees:
    print(num)
    ### --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=num,
                                min_samples_leaf=30,
                                oob_score=True,
                                random_state=2019,
                                class_weight='balanced',
                                verbose=1)
    ### --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict[num] = rf


### --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path) 

# Load model:
# rf_dict = load(model_object_path) 



# Compute OOB error per
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
oob_error_list = [None] * len(n_trees)

# Find OOB error for each forest size:
for i in range(len(n_trees)):
    oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_
else:
    # Visulaize result:
    plt.plot(n_trees, oob_error_list, 'bo',
             n_trees, oob_error_list, 'k')
    
    
    
    # Feature importance plot, modified from: 
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
top_num = 7
forest = rf_dict[500]
importances = forest.feature_importances_

# Sort in decreasing order:
indices = np.argsort(importances)[::-1]
xvarlist = X.columns[indices]

# Plot the feature importances of the forest
ax = plt.gca()
plt.title(f"Top {top_num} feature importances")
plt.bar(range(top_num), importances[indices[0:top_num]])
plt.xticks(range(top_num))
ax.set_xticklabels(xvarlist, rotation = 45, ha='right')
ax.set_xlabel("Pixel position in image")
ax.set_ylabel("Feature Importance")
plt.show()


y_pred_train = forest.predict(X)
y_pred_train[0:5]

y_pred_train_probs = pd.DataFrame(forest.predict_proba(X))
y_pred_train_probs.head()


conf_mat = confusion_matrix(y_true=y,
                            y_pred=y_pred_train)
conf_mat


class_names = list(d.values())

conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df


conf_df_pct = conf_df/conf_df.sum(axis=1)
round(conf_df_pct, 2)


preds = list(y_pred_train)

result_set = df_train
result_set['preds'] = preds
