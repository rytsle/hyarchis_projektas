import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
import joblib


# Load data from preprocess.py
from preprocess import df

# Check class distribution
class_distribution = df['sentiment'].value_counts()
print("\nClass distribution:")
print(class_distribution)

# Use Bag of Words approach with CountVectorizer
print("\nApplying TF-IDF Vectorization...")

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=7000,
    min_df=3,
    max_df=0.75,
    ngram_range=(1, 3),
    smooth_idf=True
)
X = tfidf_vectorizer.fit_transform(df['text'])

# Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print([X_train.shape, y_train.shape])

scale_pos_weight = (len(y_train) - np.sum(y_train == 0)) / np.sum(y_train == 0)


# XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',           # Multi-class classification (softmax probabilities)
    num_class=3,                          # Number of classes (positive, neutral, negative)
    max_depth=5,                          # Maximum depth of trees (controls complexity)
    learning_rate=0.05,                   # Learning rate (small value helps with precision)
    n_estimators=700,                     # Number of trees, higher can improve model but also overfitting risk
    subsample=0.85,                       # Fraction of data used to build each tree (prevent overfitting)
    colsample_bytree=0.85,                # Fraction of features used in each tree
    reg_alpha=0.1,                        # L1 regularization (helps with sparsity)
    reg_lambda=1.0,                       # L2 regularization (prevents overfitting)
    min_child_weight=3,                   # Minimum weight of a leaf node (prevents overfitting)
    eval_metric='mlogloss',               # Evaluation metric (log loss for multi-class)
    scale_pos_weight=scale_pos_weight,     # Balancing class weights (adjust if data is imbalanced)
    seed=42                               # Set random seed for reproducibility
)


xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model, vectorizer, and label encoder for future use
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


#                    precision    recall  f1-score   support

#     negative       0.77      0.53      0.63      1556
#      neutral       0.60      0.80      0.69      2223
#     positive       0.79      0.68      0.73      1717

#     accuracy                           0.68      5496
#    macro avg       0.72      0.67      0.68      5496
# weighted avg       0.71      0.68      0.68      5496