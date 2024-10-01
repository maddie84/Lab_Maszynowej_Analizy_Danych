import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1.1
df = pd.read_csv('iris.csv')

# print(df)

# # 1.2
statystyki = df.describe()

# print(statystyki)

# # 1.3 Sprawdzenie liczby brakujących wartości w każdej kolumnie
brakujace = df.isnull().sum()
# print(brakujace)

# Czy istnieją jakiekolwiek brakujące dane?
# print(df.isnull().values.any())

# 1.4 
# Wyłączenie kolumny species do normalizacji
kolumny_do_normalizacji = df.columns.drop('species')

# Utworzenie obiektu MinMaxScaler
scaler = MinMaxScaler()

# Normalizacja danych z kolumn numerycznych (bez kolumny 'species')
df[kolumny_do_normalizacji] = scaler.fit_transform(df[kolumny_do_normalizacji])

# # Wyświetlenie znormalizowanych danych
# print(df.head())

# 1.5

# # Utworzenie obiektu MinMaxScaler z zakresem [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))

# # Normalizacja danych w kolumnach numerycznych (bez kolumny 'species')
df[kolumny_do_normalizacji] = scaler.fit_transform(df[kolumny_do_normalizacji])

# # Wyświetlenie znormalizowanych danych
# print(df.head())

# 1.6

# Utworzenie obiektu StandardScaler
scaler = StandardScaler()

kolumny_do_standaryzacji = df.columns.drop('species')

# Standaryzacja danych w kolumnach numerycznych (bez kolumny 'species')
df[kolumny_do_standaryzacji] = scaler.fit_transform(df[kolumny_do_standaryzacji])

# Wyświetlenie znormalizowanych danych
# print(df.head())

# 1.7
# Zbiorcze dane pierwotne
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Pierwotne dane
axs[0, 0].scatter(df['petal_width'], df['petal_length'], c=df['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[0, 0].set_title('Dane pierwotne')
axs[0, 0].set_xlabel('Szerokość płatka [cm]')
axs[0, 0].set_ylabel('Długość płatka [cm]')

# Znormalizowane dane w zakresie [0,1]
scaler_01 = MinMaxScaler(feature_range=(0, 1))
df_01 = df.copy()
df_01[kolumny_do_normalizacji] = scaler_01.fit_transform(df_01[kolumny_do_normalizacji])

axs[0, 1].scatter(df_01['petal_width'], df_01['petal_length'], c=df_01['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[0, 1].set_title('Znormalizowane w zakresie [0, 1]')
axs[0, 1].set_xlabel('Szerokość płatka [cm]')
axs[0, 1].set_ylabel('Długość płatka [cm]')

# Znormalizowane dane w zakresie [-1,1]
scaler_neg_1_1 = MinMaxScaler(feature_range=(-1, 1))
df_neg_1_1 = df.copy()
df_neg_1_1[kolumny_do_normalizacji] = scaler_neg_1_1.fit_transform(df_neg_1_1[kolumny_do_normalizacji])

axs[1, 0].scatter(df_neg_1_1['petal_width'], df_neg_1_1['petal_length'], c=df_neg_1_1['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[1, 0].set_title('Znormalizowane w zakresie [-1, 1]')
axs[1, 0].set_xlabel('Szerokość płatka [cm]')
axs[1, 0].set_ylabel('Długość płatka [cm]')

# Standaryzowane dane
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[kolumny_do_normalizacji] = scaler.fit_transform(df_standardized[kolumny_do_normalizacji])

axs[1, 1].scatter(df_standardized['petal_width'], df_standardized['petal_length'], c=df_standardized['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[1, 1].set_title('Standaryzowane dane')
axs[1, 1].set_xlabel('Szerokość płatka [cm]')
axs[1, 1].set_ylabel('Długość płatka [cm]')

# Ustawienia layoutu
plt.tight_layout()
plt.show()

# 1.8
# Tworzenie figury z 4 wykresami
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Dane pierwotne
axs[0, 0].scatter(df['sepal_width'], df['sepal_length'], c=df['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[0, 0].set_title('Dane pierwotne')
axs[0, 0].set_xlabel('Szerokość kielicha [cm]')
axs[0, 0].set_ylabel('Długość kielicha [cm]')

# Znormalizowane dane w zakresie [0, 1]
scaler_01 = MinMaxScaler(feature_range=(0, 1))
df_01 = df.copy()
df_01[kolumny_do_normalizacji] = scaler_01.fit_transform(df_01[kolumny_do_normalizacji])

axs[0, 1].scatter(df_01['sepal_width'], df_01['sepal_length'], c=df_01['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[0, 1].set_title('Znormalizowane w zakresie [0, 1]')
axs[0, 1].set_xlabel('Szerokość kielicha [cm]')
axs[0, 1].set_ylabel('Długość kielicha [cm]')

# Znormalizowane dane w zakresie [-1, 1]
scaler_neg_1_1 = MinMaxScaler(feature_range=(-1, 1))
df_neg_1_1 = df.copy()
df_neg_1_1[kolumny_do_normalizacji] = scaler_neg_1_1.fit_transform(df_neg_1_1[kolumny_do_normalizacji])

axs[1, 0].scatter(df_neg_1_1['sepal_width'], df_neg_1_1['sepal_length'], c=df_neg_1_1['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[1, 0].set_title('Znormalizowane w zakresie [-1, 1]')
axs[1, 0].set_xlabel('Szerokość kielicha [cm]')
axs[1, 0].set_ylabel('Długość kielicha [cm]')

# Standaryzowane dane
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[kolumny_do_normalizacji] = scaler.fit_transform(df_standardized[kolumny_do_normalizacji])

axs[1, 1].scatter(df_standardized['sepal_width'], df_standardized['sepal_length'], c=df_standardized['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
axs[1, 1].set_title('Standaryzowane dane')
axs[1, 1].set_xlabel('Szerokość kielicha [cm]')
axs[1, 1].set_ylabel('Długość kielicha [cm]')

# Ustawienia layoutu
plt.tight_layout()
plt.show()
