# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import graphviz
# %matplotlib inline
from sklearn import tree
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# Load the data
df = pd.read_csv('UCI_Credit_Card.csv')

df.shape

df.head()

df.info()

df.describe()

# Renaming columns with more suitable labels
df = df.rename(columns={'default.payment.next.month': 'default', 'PAY_0': 'PAY_1'})

df.drop('ID', axis = 1, inplace =True)

df.head()

# df['SEX'].value_counts()

# Count of the occurrences of each category in the SEX variable
sex_counts = df['SEX'].value_counts()
print("\nGender Distribution:")
print("-------------------")
print(f"Male:   {sex_counts[1]:,}")
print(f"Female: {sex_counts[2]:,}")
print("-------------------")

# Plotting the distribution of SEX
plt.figure(figsize=(6, 4))
sex_counts.index = ['Female', 'Male']
sex_counts.plot(kind='bar', color=['blue', 'red'], alpha=0.9)
plt.title("Distribution of Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

df['EDUCATION'].value_counts()

fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
df.loc[fil, 'EDUCATION'] = 4
df['EDUCATION'].value_counts()

education_counts = df['EDUCATION'].value_counts()
education_counts = education_counts.loc[[1, 2, 3, 4]]
education_counts.index = ['Graduate School', 'University', 'High School', 'Others']
print("\nEducation Level Distribution:")
print("--------------------------")
print(education_counts)
print("--------------------------")

# Plotting the distribution of EDUCATION
plt.figure(figsize=(6, 4))
education_counts.plot(kind='bar', color=['green', 'blue', 'orange', 'red'], alpha=0.9)
plt.title('Distribution of Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df['MARRIAGE'].value_counts()

fil = (df.MARRIAGE == 0)
df.loc[fil, 'MARRIAGE'] = 3
df.MARRIAGE.value_counts()

# Count of the occurrences of each category in the MARRIAGE variable
marriage_counts = df['MARRIAGE'].value_counts()
marriage_counts = marriage_counts.loc[[1, 2, 3]]  # Assuming 1, 2, 3 correspond to the categories as labeled
marriage_counts.index = ['Married', 'Single', 'Others']
print("\nMarital Status Distribution:")
print("-------------------------")
print(marriage_counts)
print("-------------------------")

# Plotting the distribution of MARRIAGE
plt.figure(figsize=(6, 4))
marriage_counts.plot(kind='bar', color=['pink', 'orange', 'black'], alpha=0.9)
plt.title('Distribution of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Count of default payment status
target_counts = df['default'].value_counts()
target_counts = target_counts.loc[[0, 1]]
target_counts.index = ['No Default', 'Default']
print("\nDefault Payment Distribution:")
print("--------------------------")
print(f"No Default:  {target_counts['No Default']:,}")
print(f"Default:     {target_counts['Default']:,}")
print("--------------------------")

plt.figure(figsize=(6, 4))
target_counts.plot(kind='bar', color=['green', 'red'], alpha=0.9)
plt.title('Distribution of Default Payment Status')
plt.xlabel('Default Payment Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nCredit Limit Summary Statistics:")
print("--------------------------------")
stats = df['LIMIT_BAL'].describe()
formatted_stats = {stat: f"{value:,.0f}" for stat, value in stats.items()}
for stat, value in formatted_stats.items():
    print(f"{stat.capitalize()}: {value}")

# LIMIT_BAL Distribution
# sns.kdeplot(df['LIMIT_BAL'], color='red', linewidth=2)
plt.figure(figsize=(6, 4))
sns.histplot(df['LIMIT_BAL'], kde=True, color='green', bins=30, alpha=0.25)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xticks(rotation=0)
plt.title('Distribution of Credit Limit Balance')
plt.xlabel('Credit Limit (NT$)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nAge Summary Statistics:")
print("----------------------")
stats = df['AGE'].describe()
formatted_stats = {stat: f"{value:,.0f}" for stat, value in stats.items()}  # Format the statistics for better readability
for stat, value in formatted_stats.items():
    print(f"{stat.capitalize()}: {value}")

# AGE Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['AGE'], kde=True, color='blue', bins=20, alpha=0.25)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))  # Formatting x-axis labels for clarity
plt.xticks(rotation=0)
plt.title('Distribution of Age')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Slightly reduced alpha for subtler grid lines
plt.tight_layout()
plt.show()

# Engineering feature - Average Bill Amount
df['AVG_BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)

# Descriptive statistics for Average Bill Amount
df['AVG_BILL_AMT'].describe()

# Histogram of Average Bill Amount
plt.figure(figsize=(6, 4))
plt.hist(df['AVG_BILL_AMT'], bins=200, color='blue', alpha=0.9, range=[-10000, 200000])
plt.title('Histogram of Average Bill Amount (in range -10000 to 200000)')
plt.xlabel('Average Bill Amount')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Average Bill Amount
plt.figure(figsize=(6, 4))
plt.boxplot(df['AVG_BILL_AMT'], vert=False, flierprops=dict(markerfacecolor='g', marker='o'))
plt.title('Boxplot of Average Bill Amount (in range -10000 to 200000)')
plt.xlabel('Average Bill Amount')
plt.xlim(-10000, 200000)
plt.grid(True)
plt.show()

# Engineering feature - Average Payment Amount
df['AVG_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

# Descriptive statistics for Average Payment Amount
df['AVG_PAY_AMT'].describe()

# Histogram of Average Payment Amount
plt.figure(figsize=(6, 4))
plt.hist(df['AVG_PAY_AMT'], bins=100, color='blue', alpha=0.9, range=[0, 20000])
plt.title('Histogram of Average Payment Amount (in range 0 to 20000)')
plt.xlabel('Average Payment Amount')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Average Payment Amount
plt.figure(figsize=(6, 4))
plt.boxplot(df['AVG_PAY_AMT'], vert=False, flierprops=dict(markerfacecolor='g', marker='o'), whis=1.5)
plt.title('Boxplot of Average Payment Amount (in range 0 to 20000)')
plt.xlabel('Average Payment Amount')
plt.xlim(0, 20000)
plt.grid(True)
plt.show()

# Engineering feature - Overall Credit Utilization Ratio
df['OVERALL_UTIL_RATIO'] = df['AVG_BILL_AMT'] / df['LIMIT_BAL']

# Descriptive statistics for Overall Credit Utilization
df['OVERALL_UTIL_RATIO'].describe()

# Histogram of Overall Credit Utilization Ratio
plt.figure(figsize=(6, 4))
plt.hist(df['OVERALL_UTIL_RATIO'], bins=200, color='blue', alpha=0.9, range=[-0.1, 1.1])
plt.title('Histogram of Overall Credit Utilization Ratio (in range 0 to 1)')
plt.xlabel('Overall Credit Utilization Ratio')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Overall Credit Utilization Ratio
plt.figure(figsize=(6, 4))
plt.boxplot(df['OVERALL_UTIL_RATIO'], vert=False, flierprops=dict(markerfacecolor='g', marker='o'))
plt.title('Boxplot of Overall Credit Utilization Ratio (in range 0 to 1)')
plt.xlabel('Overall Credit Utilization Ratio')
plt.xlim(-0.1, 1.1)
plt.grid(True)
plt.show()

# Engineering feature - Debt Growth/Reduction Over Time
df['DEBT_GROWTH'] = (df['BILL_AMT6'] - df['BILL_AMT1']) / df['LIMIT_BAL']

# Descriptive statistics for Debt Growth/Reduction Over Time
df['DEBT_GROWTH'].describe()

# Histogram of Debt Growth/Reduction Over Time
plt.figure(figsize=(6, 4))
plt.hist(df['DEBT_GROWTH'], bins=200, color='blue', alpha=0.9, range=[-1, 1])
plt.title('Histogram of Debt Growth/Reduction Over Time (in range -1 to 1)')
plt.xlabel('Debt Growth/Reduction Ratio')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Debt Growth/Reduction Over Time
plt.figure(figsize=(6, 4))
plt.boxplot(df['DEBT_GROWTH'], vert=False, flierprops=dict(markerfacecolor='g', marker='o'))
plt.title('Boxplot of Debt Growth/Reduction Over Time (in range -1 to 1)')
plt.xlabel('Debt Growth/Reduction Ratio')
plt.xlim(-1, 1)
plt.grid(True)
plt.show()

df['SEX_MARRIAGE_INTERACTION'] = 0
# Assign codes based on SEX and MARRIAGE combinations
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 1), 'SEX_MARRIAGE_INTERACTION'] = 1  # Married man
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 2), 'SEX_MARRIAGE_INTERACTION'] = 2  # Single man
df.loc[(df['SEX'] == 1) & (df['MARRIAGE'] == 3), 'SEX_MARRIAGE_INTERACTION'] = 3  # Divorced man
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 1), 'SEX_MARRIAGE_INTERACTION'] = 4  # Married woman
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 2), 'SEX_MARRIAGE_INTERACTION'] = 5  # Single woman
df.loc[(df['SEX'] == 2) & (df['MARRIAGE'] == 3), 'SEX_MARRIAGE_INTERACTION'] = 6  # Divorced woman

# Plotting the distribution of sex_marriage_interaction categories
sex_marriage_interaction_counts = df['SEX_MARRIAGE_INTERACTION'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sex_marriage_interaction_counts.plot(kind='bar', color='teal')
plt.title('Distribution of Sex and Marriage Interaction Categories')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(ticks=range(6), labels=['Married Man', 'Single Man', 'Divorced Man', 'Married Woman', 'Single Woman', 'Divorced Woman'], rotation=45)
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharey=True)
fig.suptitle('Default Status by Sex and Marriage Interaction Categories', fontsize=16)

labels = ['Married Man', 'Single Man', 'Divorced Man', 'Married Woman', 'Single Woman', 'Divorced Woman']

for i, label in enumerate(labels, start=1):
    # Filter the dataframe for the specific category and count the default status
    category_data = df[df['SEX_MARRIAGE_INTERACTION'] == i]['default'].value_counts().sort_index()
    
    ax = axes[(i-1) // 2, (i-1) % 2]
    category_data.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title(label)
    ax.set_xlabel('Default Status (0: No, 1: Yes)')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(['No Default', 'Default'], rotation=0)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

summary = df.groupby('SEX_MARRIAGE_INTERACTION')['default'].value_counts().unstack(fill_value=0)
summary.columns = ['No Default', 'Default']
summary['Total'] = summary.sum(axis=1)

category_labels = {
    1: 'Married Man',
    2: 'Single Man',
    3: 'Divorced Man',
    4: 'Married Woman',
    5: 'Single Woman',
    6: 'Divorced Woman'
}
summary.rename(index=category_labels, inplace=True)
summary

df.columns

excluded_columns = ['default']
features = df.columns.difference(excluded_columns).tolist()

# Creating the feature and target DataFrames
X = df[features].copy()
y = df['default'].copy()

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

