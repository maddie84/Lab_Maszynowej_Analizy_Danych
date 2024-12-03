import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('co2.csv')

print(df.isnull().sum())
print(df.describe())
print(df.dtypes)

le = LabelEncoder()

df['marka_encoded'] = le.fit_transform(df['Make'])
df['model_encoded'] = le.fit_transform(df['Model'])
df['transmission_encoded'] = le.fit_transform(df['Transmission'])
df['fuel_type_encoded'] = le.fit_transform(df['Fuel Type'])
df['vehicle_class_encoded'] = le.fit_transform(df['Vehicle Class'])

df.drop(columns=['Make', 'Model', 'Transmission', 'Fuel Type','Vehicle Class'], inplace=True)

df.head()

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix for Numerical Features")
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

min_value = df['CO2 Emissions(g/km)'].min()
max_value = df['CO2 Emissions(g/km)'].max()

bins = [min_value, 200, 250, 300, max_value+1]
labels = ['niska', 'średnia', 'wysoka', 'bardzo wysoka']
df['CO2_emission_category'] = pd.cut(df['CO2 Emissions(g/km)'], bins=bins, labels=labels, right=False)
df['CO2_emission_category_encoded'] = le.fit_transform(df['CO2_emission_category'])

df.drop(columns=['CO2_emission_category'], inplace=True)
df.drop(columns=['CO2 Emissions(g/km)'], inplace=True)

df.head()
print(df.isnull().sum())

y = df['CO2_emission_category_encoded']
X = df.drop(['CO2_emission_category_encoded'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

