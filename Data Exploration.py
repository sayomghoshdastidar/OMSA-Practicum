#Importing relevant packages
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

#Loading the data
df = pd.read_csv('NFL Coaches Data Set.csv')

#Defining the response variable
response = 'fired'

#Separating the binary and numeric predictors
binary_vars = ['sb_champ', 'runner_up', 'scandal', 'minority', 'pro_bowl_qb', 'first_tenure', 'coty']
numeric_vars = ['tenure', 'win_pct', 'final_yr_win_pct']

#Creating a contingency table and running a Chi-squared test for binary predictors
print("\nBinary vs Fired: Chi-Squared Tests")

#Looping over all binary predictors and checking independence with fired
for var in binary_vars:
    print(f"\nVariable: {var}")

    #Contingency table and reporting results
    ct = pd.crosstab(df[response], df[var])
    print("Contingency Table:")
    print(ct)

    #Chi-squared test and reporting results
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}")

#Point-biserial correlation for numeric predictors
print("\n\n\nNumeric vs Fired: Point-Biserial Correlation\n")

#Looping over all numeric predictors and checking correlation with fired
for var in numeric_vars:
    corr, p = pointbiserialr(df[response], df[var])
    print(f"{var}: Correlation = {corr:.2f}, p-value = {p:.4f}")

#Checking if fired_midseason has a different relationship with minority than fired
print("\n\n\nFired_midseason vs Minority Chi-Squared Test")
ct = pd.crosstab(df['fired_midseason'], df['minority'])
print(ct)
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}")

#Testing relationship between win_pct and final_yr_win_pct
print("\n\n\nWin_pct and Final_yr_win_pct Correlation")
corr = df[['win_pct', 'final_yr_win_pct']].corr()
print(corr)
