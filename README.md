# Machine Learning Model: Feature Selection, Training, and Evaluation

## Overview
This Jupyter Notebook demonstrates the complete workflow of feature selection, training, and evaluation of a machine learning model using the Titanic dataset from Seaborn. It covers data preprocessing, model training using RandomForestClassifier, and performance evaluation.

## Steps Covered

### Step 1: Import Required Libraries
The following libraries are used in the notebook:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `matplotlib.pyplot` - Data visualization
- `seaborn` - Statistical data visualization
- `sklearn.preprocessing.LabelEncoder` - Encoding categorical variables
- `sklearn.model_selection.train_test_split` - Splitting the dataset into training and testing sets
- `sklearn.preprocessing.StandardScaler` - Standardizing numerical features
- `sklearn.ensemble.RandomForestClassifier` - Machine learning model for classification
- `sklearn.metrics` - Performance evaluation metrics (accuracy, classification report, confusion matrix)

### Step 2: Load Dataset
The Titanic dataset is loaded from Seaborn:
```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
```
This dataset contains information about passengers, such as age, gender, fare, and survival status.

### Step 3: Exploratory Data Analysis (EDA)
- Display dataset sample
- Check dataset information
- Identify missing values
```python
print("Dataset Sample:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
```

### Step 4: Data Preprocessing
- Drop rows with missing target values (`'survived'` column)
- Select features and target column
- Convert categorical features to numerical using one-hot encoding
```python
df = df.dropna(subset=['survived'])
features = df.drop(columns=['survived'])
target = df['survived']
features = pd.get_dummies(features, drop_first=True)
```

### Step 5: Split Data into Training and Testing Sets
- Train-test split (80-20 ratio)
- Ensure feature consistency between training and testing sets
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```

### Step 6: Scale Numeric Features
- Standardize numerical values using `StandardScaler`
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 7: Train Model (Random Forest Classifier)
- Train a `RandomForestClassifier` model
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Step 8: Make Predictions
- Predict survival status
```python
y_pred = model.predict(X_test)
```

### Step 9: Evaluate Model Performance
- Calculate model accuracy
- Generate a classification report
```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
**Sample Output:**
```
Model Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       105
           1       1.00      1.00      1.00        74

    accuracy                           1.00       179
   macro avg       1.00      1.00      1.00       179
weighted avg       1.00      1.00      1.00       179
```

### Step 10: Confusion Matrix Visualization
- Display confusion matrix as a heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

### Completion
```python
print("\nMachine Learning Model Training Complete!")
```

## Requirements
To run this notebook, install the required libraries if they are not already installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1. Open Jupyter Notebook and run the notebook step by step.
2. Perform Exploratory Data Analysis (EDA) to understand the dataset.
3. Preprocess the data by handling missing values and encoding categorical variables.
4. Split the dataset into training and testing sets.
5. Scale numerical features to improve model performance.
6. Train a `RandomForestClassifier` model on the dataset.
7. Evaluate the model using accuracy, classification report, and confusion matrix.
8. Visualize the confusion matrix to interpret the results.

## Output
The notebook produces:
- Data visualizations for understanding patterns
- A trained RandomForest model
- Evaluation metrics like accuracy, confusion matrix, and classification report

