#JOINTLY ANSWERING KEY QUESTIONS 1 AND 2

#Importing relevant packages
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer

#Setting random seed for reproducibility
np.random.seed(6748)

#Loading the data
df = pd.read_csv('NFL Coaches Data Set.csv')

#Filtering for inactive coaches only
df_inactive = df[df['active'] == 0].copy()

#Keeping only the selected predictors
predictors = ['sb_champ', 'pro_bowl_qb', 'coty', 'tenure', 'win_pct']
X = df_inactive[predictors]
y = df_inactive['fired']

#Setting up stratified k-fold cross-validation using 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=6748)

#Logistic Regression
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),  #Standardizing numeric predictors: tenure and win percentage
    ('clf', LogisticRegression(random_state=6748, solver='lbfgs', max_iter=1000))
])

#Reporting cross-validated accuracy
log_acc = cross_val_score(
    logistic_pipeline, X, y,
    cv=cv, scoring='accuracy'
)
print(f"Logistic Regression CV Accuracy: {log_acc.mean():.3f}")

#Reporting cross-validated AUC
log_auc = cross_val_score(
    logistic_pipeline, X, y,
    cv=cv, scoring='roc_auc'
)
print(f"Logistic Regression CV AUC: {log_auc.mean():.3f}")

#Fitting the final model on all data to get coefficients
logistic_pipeline.fit(X, y)

#Reporting coefficients and odds ratios in an interpretable manner
coefs = logistic_pipeline.named_steps['clf'].coef_[0]

#Calculating standard deviations for numeric predictors
#This will be necessary for interpretation since we have scaled the numeric predictors
numeric_predictors = ['tenure', 'win_pct']
sd_dict = {}
for var in numeric_predictors:
    sd = X[var].std(ddof=0)
    sd_dict[var] = sd

#Printing interpretations
#Since numeric and binary predictors require different interpretations, they are separated into two code blocks
print("\n\nLogistic Regression Coefficients and Odds Ratios Interpreted:\n")
for var, coef in zip(predictors, coefs):
    odds_ratio = np.exp(coef)
    percent_change = (odds_ratio - 1) * 100
    if var in numeric_predictors:
        sd = sd_dict[var]
        print(f"{var}:")
        print(f"Coefficient: {coef:.3f}")
        print(f"Odds ratio: {odds_ratio:.3f}")
        print(f"Original standard deviation before scaling: {sd:.3f}")
        print(f"This means that adding 1 standard deviation of {sd:.3f} units leads to a {percent_change:.1f}% change in predicted firing probability\n")
    else:
        print(f"{var}:")
        print(f"Coefficient: {coef:.3f}")
        print(f"Odds ratio: {odds_ratio:.3f}")
        print(f"This means that going from 0 to 1 leads to a {percent_change:.1f}% change in predicted firing probability\n")

#Random Forest with Grid Search for hyperparameter tuning
rf = RandomForestClassifier(random_state=6748, criterion='gini')

#Only one entry for each because these were found to be the best parameter values in earlier runs
param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_leaf': [4]
}

#Calculating both accuracy and AUC metrics
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'AUC': 'roc_auc'
}

#Performing grid search to tune hyperparameters
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=cv,
    scoring=scoring,
    refit='AUC',  #Choosing best model by AUC, already found to be 100 estimators, no max depth, and 4 samples per leaf minimum
    return_train_score=True
)

#Fitting the models
grid_search.fit(X, y)

#Reporting the results
print(f"\n\nRandom Forest Best Params: {grid_search.best_params_}")
print(f"Random Forest CV Accuracy: {grid_search.cv_results_['mean_test_Accuracy'][grid_search.best_index_]:.3f}")
print(f"Random Forest CV AUC: {grid_search.best_score_:.3f}")

#Fitting the best random forest on all the data
best_rf = grid_search.best_estimator_
best_rf.fit(X, y)

#Ranking variable importance by using permutation importance
result = permutation_importance(
    best_rf, X, y,
    n_repeats=50,
    random_state=6748
)

#Reporting the permutation importance with standard deviation
print("\nRandom Forest Permutation Importances:")
for i in result.importances_mean.argsort()[::-1]:
    print(f"{predictors[i]}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")



#KEY QUESTION 3

#Importing package to get cross-validated predictions so I am not predicting on the same data that I have trained with
#Generates out-of-sample predictions for every observation
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt

#Logistic Regression cross-validation predicted firing probabilities
log_probs_cv = cross_val_predict(
    logistic_pipeline, X, y,
    cv=cv, method='predict_proba'
)[:, 1]

#Random Forest cross-validation predicted firing probabilities
rf_probs_cv = cross_val_predict(
    rf, X, y,
    cv=cv, method='predict_proba'
)[:, 1]

#Adding cross-validated probabilities to the dataframe
df_preds_cv = df_inactive.copy()
df_preds_cv['log_prob_fired_cv'] = log_probs_cv
df_preds_cv['rf_prob_fired_cv'] = rf_probs_cv

#Reporting surprise firings -- fired = 1 but predicted firing probability is low
print("\nLogistic Regression Surprise Firings:")
log_surprise_firing = df_preds_cv[df_preds_cv['fired'] == 1].sort_values('log_prob_fired_cv').head(10).reset_index(drop=True)
print(log_surprise_firing[['name', 'team', 'win_pct', 'tenure', 'log_prob_fired_cv']].round(2))

print("\nRandom Forest Surprise Firings:")
rf_surprise_firing = df_preds_cv[df_preds_cv['fired'] == 1].sort_values('rf_prob_fired_cv').head(10).reset_index(drop=True)
print(rf_surprise_firing[['name', 'team', 'win_pct', 'tenure', 'rf_prob_fired_cv']].round(2))

#Reporting unexpected survivors -- fired = 0 but predicted firing probability is high
print("\nLogistic Regression Unexpected Survivors:")
log_survivors = df_preds_cv[df_preds_cv['fired'] == 0].sort_values('log_prob_fired_cv', ascending=False).head(5).reset_index(drop=True)
print(log_survivors[['name', 'team', 'win_pct', 'tenure', 'log_prob_fired_cv']].round(2))

print("\nRandom Forest Unexpected Survivors:")
rf_survivors = df_preds_cv[df_preds_cv['fired'] == 0].sort_values('rf_prob_fired_cv', ascending=False).head(5).reset_index(drop=True)
print(rf_survivors[['name', 'team', 'win_pct', 'tenure', 'rf_prob_fired_cv']].round(2))

#Reformatting surprise firings and unexpected survivors to put into a plot for logistic regression
log_surprise_firing['type'] = 'Surprise Firing'
log_survivors['type'] = 'Unexpected Survivor'

#Putting them in one dataframe
log_dotplot_df = pd.concat([log_surprise_firing, log_survivors])

#Adding a column for coach label with name and team
log_dotplot_df['label'] = log_dotplot_df['name'] + " (" + log_dotplot_df['team'] + ")"

#Manually offsetting a few coach labels for better visualization
#Labels were overlapping
log_dotplot_df['label_x_offset'] = 0.0
log_dotplot_df['label_y_offset'] = 0.0

#Coach labels to be moved
coaches_to_move = ['Jon Gruden (LV)', 'Nick Saban (MIA)', 'Doug Pederson (PHI)', 'Mike Shanahan (DEN)']
for coach in coaches_to_move:
    log_dotplot_df.loc[log_dotplot_df['label'] == coach, ['label_x_offset', 'label_y_offset']] = [-0.050, -0.015]

#Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=log_dotplot_df,
    x='win_pct',
    y='log_prob_fired_cv',
    hue='type',
    s=80,
    alpha=0.8,
    palette={'Surprise Firing': 'steelblue', 'Unexpected Survivor': 'darkred'}
)

#Adding text labels near each point
for _, row in log_dotplot_df.iterrows():
    plt.text(
        row['win_pct'] + 0.005 + row['label_x_offset'], #Adding an offset
        row['log_prob_fired_cv'] + row['label_y_offset'], #Adding an offset
        row['label'],
        fontsize=8
    )

#Labeling
plt.xlabel('Win Percentage')
plt.ylabel('Predicted Firing Probability')
plt.title('Logistic Regression Top 10 Surprise Firings and Top 5 Unexpected Survivors')

#Manually extending x-axis limits
plt.xlim(left=0.15, right=0.8)

#Putting legend inside plot in the upper right corner
plt.legend(title='Type', loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
plt.tight_layout()
plt.show()

#Repeating same plotting procedure for the random forest
rf_surprise_firing['type'] = 'Surprise Firing'
rf_survivors['type'] = 'Unexpected Survivor'

#Combining into one dataframe
rf_dotplot_df = pd.concat([rf_surprise_firing, rf_survivors])

#Creating label column
rf_dotplot_df['label'] = rf_dotplot_df['name'] + " (" + rf_dotplot_df['team'] + ")"

#Offsetting names that overlap
rf_dotplot_df['label_x_offset'] = 0.0
rf_dotplot_df['label_y_offset'] = 0.0

#Manual offset for one coach whose label is overlapping with another
rf_dotplot_df.loc[
    rf_dotplot_df['label'] == 'Tony Sparano (MIA)',
    ['label_x_offset', 'label_y_offset']
] = [-0.05, -0.025]

#Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=rf_dotplot_df,
    x='win_pct',
    y='rf_prob_fired_cv',
    hue='type',
    s=80,
    alpha=0.8,
    palette={'Surprise Firing': 'steelblue', 'Unexpected Survivor': 'darkred'}
)

#Adding text labels for points with coach's name and team
for _, row in rf_dotplot_df.iterrows():
    plt.text(
        row['win_pct'] + 0.005 + row['label_x_offset'],
        row['rf_prob_fired_cv'] + row['label_y_offset'],
        row['label'],
        fontsize=8
    )

#Labeling
plt.xlabel('Win Percentage')
plt.ylabel('Predicted Firing Probability')
plt.title('Random Forest Top 10 Surprise Firings and Top 5 Unexpected Survivors')

#Extending x-axis manually for better spacing
plt.xlim(left=0.15, right=0.8)

#Once again moving legend to top right corner of plot
plt.legend(title='Type', loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
plt.tight_layout()
plt.show()



#KEY QUESTION 4

#Filtering for active coaches only
df_active = df[df['active'] == 1].copy()
X_active = df_active[predictors]

#Logistic Regression predictions on active coaches
#Using the created pipeline applies the same preprocessing steps, so scaling with happen again for tenure and win_pct
log_active_probs = logistic_pipeline.predict_proba(X_active)[:, 1]

#Random Forest predictions on active coaches
rf_active_probs = best_rf.predict_proba(X_active)[:, 1]

#Adding the predictions to the active DataFrame
df_active['log_prob_fired'] = log_active_probs
df_active['rf_prob_fired'] = rf_active_probs

#Printing the top 5 most at-risk coaches and the top 5 safest coaches
print("\nLogistic Regression Most At-Risk Coaches:")
print(df_active[['name', 'team', 'win_pct', 'tenure', 'log_prob_fired']].round(2).sort_values('log_prob_fired', ascending=False).head(5).reset_index(drop=True))

print("\nRandom Forest Most At-Risk Coaches:")
print(df_active[['name', 'team', 'win_pct', 'tenure', 'rf_prob_fired']].round(2).sort_values('rf_prob_fired', ascending=False).head(5).reset_index(drop=True))

print("\nLogistic Regression Safest Coaches:")
print(df_active[['name', 'team', 'win_pct', 'tenure', 'log_prob_fired']].round(2).sort_values('log_prob_fired').head(5).reset_index(drop=True))

print("\nRandom Forest Safest Coaches:")
print(df_active[['name', 'team', 'win_pct', 'tenure', 'rf_prob_fired']].round(2).sort_values('rf_prob_fired').head(5).reset_index(drop=True))

