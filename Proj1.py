import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("bank-full.csv", sep=';')  # Use ';' as separator
print("Original shape:", df.shape)

# -----------------------------
# 2. Drop missing values
# -----------------------------
df.dropna(inplace=True)
print("After dropping NAs:", df.shape)

# -----------------------------
# 3. Encode categorical variables
# -----------------------------
categorical_cols = ['job', 'marital', 'education', 'default',
                    'housing', 'loan', 'contact', 'month',
                    'poutcome', 'y']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for possible inverse transform

# -----------------------------
# 4. Define features and target
# -----------------------------
X = df.drop(columns=['y'])  # Features
y = df['y']                 # Target

print("Feature shape (X):", X.shape)
print("Target distribution:\n", y.value_counts())

# -----------------------------
# 5. Train/test splits & evaluation
# -----------------------------
# Generate train/test split ratios from 90/10 to 10/90
split_ratios = [i / 10 for i in range(9, 0, -1)]

for ratio in split_ratios:
    print(f"\n=== Train/Test Split: {int(ratio*100)}/{int((1 - ratio)*100)} ===")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - ratio, random_state=42)

    # Train the Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("The accuracy of my Naive Bayes Model is:", accuracy)
    print("The F1 Score of my Naive Bayes Model is:", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix (Train/Test Split: {int(ratio*100)}/{int((1 - ratio)*100)})")
    plt.show()