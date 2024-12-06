import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the CSV file
df = pd.read_csv('Classification_Ranking.csv', sep=';')

# Preprocess the data: map labels to binary values
df['Label'] = df['Label'].map({'R': 1, 'NR': 0})

# Specify which columns need conversion from string to float
columns_to_convert = [
    'abstract_Readability', 'Method_Readability', 'Results_Readability',
    'Conclusion_Readability', 'Avg_Readability', 'Avg_Certainty',
    'Abstract_Certainty', 'Method_Certainty', 'Result_Certainty',
    'Conclusion_Certainty'
]

# Replace commas with dots and convert to numeric, handling errors
for column in columns_to_convert:
    df[column] = df[column].str.replace(',', '.').astype(float)

# Handle NaN values
df.fillna(0, inplace=True)

# Separate features and labels
features = df[columns_to_convert]
labels = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to hold models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Classifier": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB()
}

# Initialize a list to store results and a dictionary for feature importance aggregation
results = []
feature_importances = {feature: 0 for feature in features.columns}  # Initialize importance dict

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Predictions
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Store results in the list
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1 Score': f1,
    })

    # Display results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Feature importance or coefficients (if applicable)
    if name == "Logistic Regression":
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({'Feature': features.columns, 'Coefficient': coefficients})
        top_coef_df = coef_df.sort_values(by='Coefficient', ascending=False).head(5)
        print("Top 5 Coefficients (Logistic Regression):")
        print(top_coef_df)

        # Aggregate coefficients for ranking
        for i, feature in enumerate(features.columns):
            feature_importances[feature] += abs(coefficients[i])

    elif name in ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree"]:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
        top_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)
        print(f"Top 5 Feature Importances ({name}):")
        print(top_importance_df)

        # Aggregate importances
        for i, feature in enumerate(features.columns):
            feature_importances[feature] += importances[i]

    elif name == "Support Vector Classifier":
        # Since SVC does not have feature importances, we skip this section.
        print("Support Vector Classifier does not provide feature importances.")

# Convert results to DataFrame for sorting
results_df = pd.DataFrame(results)

# Sort the results by Accuracy in descending order
sorted_results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# Display the sorted results
print("\nModel Ranking based on Accuracy:")
print(sorted_results_df)

# Create a DataFrame for feature importances and sort it
features_importances_df = pd.DataFrame(feature_importances.items(), columns=['Feature', 'Total Importance'])
sorted_importances_df = features_importances_df.sort_values(by='Total Importance', ascending=False).reset_index(drop=True)

print("\nFeature Importance Ranking:")
print(sorted_importances_df)
