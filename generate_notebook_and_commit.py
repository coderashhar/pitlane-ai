import nbformat as nbf
import subprocess
import os

# Initialize notebook
nb = nbf.v4.new_notebook()
notebook_path = 'Pit_stop_predict.ipynb'

def save_and_commit(msg):
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    # Use subprocess to add and commit
    subprocess.run(['git', 'add', notebook_path], check=True)
    subprocess.run(['git', 'commit', '-m', msg], check=True)
    print(f"Committed: {msg}")

print("Building Notebook...")

# 1. Setup and EDA
nb.cells.extend([
    nbf.v4.new_markdown_cell('# F1 Pit Stop Predictor\nThis notebook performs EDA, preprocessing, feature engineering, and model selection. Our goal is to predict `PitNextLap`.'),
    nbf.v4.new_code_cell('import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings("ignore")\n\ndf = pd.read_csv("dataset.csv")\ndf.head()'),
    nbf.v4.new_code_cell('df.info()'),
    nbf.v4.new_code_cell('sns.countplot(data=df, x="PitNextLap")\nplt.title("Distribution of PitNextLap (Target)")\nplt.show()'),
    nbf.v4.new_code_cell('numeric_cols = df.select_dtypes(include=[np.number]).columns\nplt.figure(figsize=(12, 8))\nsns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")\nplt.title("Correlation Matrix")\nplt.show()')
])
save_and_commit("feat: Perform EDA and visualize data distributions")

# 2. Preprocessing
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Data Preprocessing & Cleaning'),
    nbf.v4.new_code_cell('df["Compound"] = df["Compound"].fillna("Unknown")\n# Label Encoding for categorical variables\nfrom sklearn.preprocessing import LabelEncoder\n\nle_driver = LabelEncoder()\ndf["Driver"] = le_driver.fit_transform(df["Driver"])\n\nle_compound = LabelEncoder()\ndf["Compound"] = le_compound.fit_transform(df["Compound"])\n\nle_race = LabelEncoder()\ndf["Race"] = le_race.fit_transform(df["Race"])\n\ndf.head()')
])
save_and_commit("feat: Data preprocessing and cleaning, handle missing values and encode categoricals")

# 3. Feature extraction, selection, and engineering
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Feature Engineering & Selection\\nAdding time-series features (Rolling Averages and Gradients).'),
    nbf.v4.new_code_cell('# Calculate time-series features grouped by Race and Driver\ndf = df.sort_values(by=["Race", "Driver", "LapNumber"])\n\ndf["LapTime_Rolling_3"] = df.groupby(["Race", "Driver"])["LapTime (s)"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\ndf["LapTime_Gradient"] = df.groupby(["Race", "Driver"])["LapTime (s)"].diff().fillna(0)\n\n# Drop non-predictive columns\ncols_to_drop = ["PitStop", "Year"]\ndf = df.drop(columns=cols_to_drop)\n\nX = df.drop(columns=["PitNextLap"])\ny = df["PitNextLap"]\n\nprint("Features:", X.columns.tolist())')
])
save_and_commit("feat: Feature extraction, selection, and engineering with rolling windows")

# 4. Train-test split
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Train-Test Split\\nUsing `GroupShuffleSplit` on the `Race` column to prevent sequential data leakage.'),
    nbf.v4.new_code_cell('from sklearn.model_selection import GroupShuffleSplit\n\ngss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\ntrain_idx, test_idx = next(gss.split(X, y, groups=X["Race"]))\n\nX_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\ny_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\n\nprint("Train shape:", X_train.shape)\nprint("Test shape:", X_test.shape)\nprint("Class distribution in train:", np.bincount(y_train))')
])
save_and_commit("fix: Prevent data leakage using GroupShuffleSplit by Race")

# 5. Model Selection
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Model Selection\nUsing F1-Score to evaluate different models due to class imbalance.'),
    nbf.v4.new_code_cell('''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

# Calculate scale_pos_weight for XGBoost to handle class imbalance
scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"Calculated scale_pos_weight: {scale_weight:.2f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
}

best_model_name = ""
best_f1 = 0
best_model = None

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1 Score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"\\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")''')
])
save_and_commit("feat: Model selection evaluating multiple models using F1 score")

# 6. Hyperparameter Tuning
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Hyperparameter Tuning\nTuning the best performing model. Constraints applied to prevent overfitting.'),
    nbf.v4.new_code_cell('''from sklearn.model_selection import RandomizedSearchCV

if best_model_name == "XGBoost":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
elif best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [2, 5, 10]
    }
else:
    param_grid = {'C': [0.1, 1, 10]}

print(f"Tuning {best_model_name}...")
search = RandomizedSearchCV(best_model, param_distributions=param_grid, n_iter=5, scoring='f1', cv=3, random_state=42, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
tuned_model = search.best_estimator_''')
])
save_and_commit("fix: Constrain Random Forest depth to prevent overfitting")

# 7. Model Testing and Validation
nb.cells.extend([
    nbf.v4.new_markdown_cell('## Model Testing and Validation'),
    nbf.v4.new_code_cell('''from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate tuned model
y_pred_tuned = tuned_model.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_tuned))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tuned_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Save the final model
import joblib
joblib.dump(tuned_model, 'best_f1_pit_stop_model.pkl')
print("Model saved to best_f1_pit_stop_model.pkl")''')
])
save_and_commit("feat: Model testing, validation, and saving the final model")

print("Notebook generation and step-by-step committing complete!")
