#Importing relevant packages
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#List of predictors
predictors = [
    'tenure',
    'win_pct',
    'final_yr_win_pct',
    'sb_champ',
    'runner_up',
    'scandal',
    'minority',
    'pro_bowl_qb',
    'first_tenure',
    'coty'
]

#Loading data
df = pd.read_csv('NFL Coaches Data Set.csv')

#Creating matrix of predictors
X = df[predictors]
X = add_constant(X)

#Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

#Reporting results
print('\n VIF for all predictors:')
print(vif_data)
