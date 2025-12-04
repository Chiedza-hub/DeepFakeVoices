
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

# Load your DataFrame
df = pd.read_csv("features.csv")

# Separate features and labels and drop features related to F0 (did not calculate well)
X = df.drop(columns=["label", "filename", "path", "ext", "f0mean", "f0std", "f0min", "f0max", "f0range", "voicedratio", "duration_sec"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# calculate mutual information for each feature

from sklearn.feature_selection import mutual_info_classif

# X_train = feature matrix for training
# y_train = labels for training
mi_scores = mutual_info_classif(X_train, y_train, discrete_features=False)



# Convert to DataFrame and sort
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': mi_scores
}).sort_values(by='MI Score', ascending=False)

# Select top 20 features
top_20 = mi_df.head(20)

# Plot
plt.figure(figsize=(10, 8))
plt.barh(top_20['Feature'], top_20['MI Score'], color='skyblue')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title('Top 20 Features by Mutual Information')
plt.gca().invert_yaxis()  # So highest MI appears at the top
plt.show()

