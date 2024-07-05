import seaborn as sns

# Load Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())
# Check for missing values
print(titanic.isnull().sum())

# Handling missing values: Age and Embarked
# Replace missing Age values with the median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Replace missing Embarked values with the mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop Cabin column due to high number of missing values
titanic.drop('deck', axis=1, inplace=True)

# Confirm no more missing values
print(titanic.isnull().sum())
# Basic statistics
print(titanic.describe())

# Count of survivors
print(titanic['survived'].value_counts())
import matplotlib.pyplot as plt

# Relationship between Pclass and Survived
sns.barplot(x='pclass', y='survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival distribution by Age
sns.histplot(x='age', hue='survived', data=titanic, kde=True)
plt.title('Survival Distribution by Age')
plt.show()

# Survival by Sex
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival Count by Sex')
plt.show()

# Survival by Embarked Port
sns.countplot(x='embarked', hue='survived', data=titanic)
plt.title('Survival Count by Embarked Port')
plt.show()

# Fare distribution by Pclass
sns.boxplot(x='pclass', y='fare', data=titanic)
plt.title('Fare Distribution by Passenger Class')
plt.show()