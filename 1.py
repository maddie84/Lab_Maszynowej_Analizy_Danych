import lime
from lime.lime_tabular import LimeTabularExplainer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=200 / len(df), random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Raport z klasyfikacji:\n", class_report)

print("\nMacierz błędów:")
print(conf_matrix)

explainer = LimeTabularExplainer(
        X_train.values,
        mode="classification",
        feature_names=X.columns,
        class_names=["Did not Survive", "Survived"],
        discretize_continuous=True,
        random_state=42
    )

explainer

# Znajdowanie przypadków TN, FP, FN, TP
y_test = y_test.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)

true_negative_indices = (y_test == 0) & (y_pred == 0)
false_positive_indices = (y_test == 0) & (y_pred == 1)
false_negative_indices = (y_test == 1) & (y_pred == 0)
true_positive_indices = (y_test == 1) & (y_pred == 1)

tn_index = true_negative_indices[true_negative_indices].index[0]
fp_index = false_positive_indices[false_positive_indices].index[0]
fn_index = false_negative_indices[false_negative_indices].index[0]
tp_index = true_positive_indices[true_positive_indices].index[0]

lime_results = {}

for case, index in zip(["TN", "FP", "FN", "TP"], [tn_index, fp_index, fn_index, tp_index]):
    explanation = explainer.explain_instance(
        X_test.iloc[index].values,
        model.predict_proba,
        num_features=5
    )
    lime_results[case] = {
        "index": index,
        "actual": y_test[index],
        "predicted": y_pred[index],
        "explanation": explanation.as_list()
    }

for case, result in lime_results.items():
    print(f"Case: {case}")
    print(f"Index: {result['index']}")
    print(f"Actual Label: {result['actual']}")
    print(f"Predicted Label: {result['predicted']}")
    print("Explanation:")
    for feature, weight in result["explanation"]:
        print(f"  {feature}: {weight}")
    print("\n")

