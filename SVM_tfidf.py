from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocess import df
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=7000)

X = vectorizer.fit_transform(df['text'])

# Encode sentiment labels with LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(df['sentiment'])
print(f"Classes: {label_encoder.classes_}")
print(f"Encoded labels: {label_encoder.transform(['negative', 'neutral', 'positive'])}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

svm_model = SVC(
    kernel='rbf',                # Kernel type (linear, poly, rbf, sigmoid)
    class_weight='balanced',  # Adjust for class imbalance
    random_state=42          # Reproducibility
)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model, vectorizer, and label encoder for future use
import joblib   
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

#                 precision    recall  f1-score   support

#     negative       0.76      0.55      0.63      1572
#      neutral       0.61      0.80      0.70      2236
#     positive       0.81      0.69      0.75      1688

#     accuracy                           0.70      5496
#    macro avg       0.73      0.68      0.69      5496
# weighted avg       0.72      0.70      0.69      5496

# Negative Class: Needs better recall. Consider oversampling negative cases or adjusting class weights.

# Neutral Class: Precision can be improved by reducing over-prediction.

# Positive Class: High precision but moderate recall indicates a need for better recall tuning.
