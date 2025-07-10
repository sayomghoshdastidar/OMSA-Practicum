#Importing relevant packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading data
df = pd.read_csv('NFL Coaches Data Set.csv')

#Filtering only for coaches who were fired
df_inactive = df[df['active'] == 0].copy()
fired_coaches = df_inactive[df_inactive['fired'] == 1]

#Creating the histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    data=fired_coaches,
    x='tenure',
    bins=14,
    kde=True,
    color='steelblue',
    edgecolor='black'
)

#Labeling and plotting
plt.xlabel('Tenure Length (Years)')
plt.ylabel('Frequency')
plt.title('Distribution of Tenure Length for Fired Coaches')
plt.tight_layout()
plt.show()
