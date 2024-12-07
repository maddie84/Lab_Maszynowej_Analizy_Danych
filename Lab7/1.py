df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'Message'])

X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Przeprowadzenie transformacji danych tekstowych przy użyciu TF-IDF vectorizer.

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Raport z klasyfikacji:\n", class_report)

print("\nMacierz błędów:")
print(conf_matrix)

# Wyjaśnienie klasyfikacji tekstu za pomocą algorytmu LIME 
#Utworzenie wyjaśnienia przy pomocy klasy LimeTextExplainer, dopasowując explainer do danych treningowych.

explainer = LimeTextExplainer(class_names=model.classes_)

## 3. Wyjaśnienie lokalnie wyników dla 4 różnych przypadków klasyfikacji wyświetlenie etykiety rzeczywistej,  
# etykiety  przewidzianej  oraz  indeks  obserwacji:  True  Negative  (TN),  False Positive (FP), False Negative (FN),
#  i True Positive (TP), uwzględniając prawdopodobieństwo i cechy mające największy wpływ na predykcję.

## zrob mi to tak ladnie za pomoca algorytmu lime jak w innym zadanaich zrobiles