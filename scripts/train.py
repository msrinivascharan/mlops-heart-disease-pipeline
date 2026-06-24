import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

       # Load clean data
df = pd.read_csv('data/heart_clean.csv')

       # Features and target
X = df.drop('target', axis=1)
y = df['target']

       # Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42
       )

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

       # Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

       # Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
             target_names=['No Disease', 'Disease']))

       # Save model to disk
with open('models/heart_model.pkl', 'wb') as f:
           pickle.dump(model, f)
print("\nModel saved to models/heart_model.pkl")

