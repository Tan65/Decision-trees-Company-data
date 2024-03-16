# Decision-trees-Company-data
Problem Statement: A cloth manufacturing company is interested to know about the segment or attributes causes high sale.  Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) &amp; all other variable will be independent in the analysis.  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Company_Data.csv')

# Univariate analysis
numeric_vars = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age']
categorical_vars = ['ShelveLoc', 'Education', 'Urban', 'US']

# Histograms for numerical variables
for var in numeric_vars:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[var], kde=True, bins=20)
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# Bar plots for categorical variables
for var in categorical_vars:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=var)
    plt.title(f'Bar plot of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# Bivariate analysis
# Scatter plots for numerical variables vs target variable
for var in numeric_vars:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=var, y='Sales')
    plt.title(f'Scatter plot of Sales vs {var}')
    plt.xlabel(var)
    plt.ylabel('Sales')
    plt.show()

# Box plots for categorical variables vs target variable
for var in categorical_vars:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=var, y='Sales')
    plt.title(f'Box plot of Sales vs {var}')
    plt.xlabel(var)
    plt.ylabel('Sales')
    plt.show()

# Multivariate analysis
sns.pairplot(df, vars=numeric_vars, kind='scatter', hue='ShelveLoc')
plt.suptitle('Pair Plot of Numerical Variables with ShelveLoc')
plt.show()

# Correlation analysis
correlation_matrix = df[numeric_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
