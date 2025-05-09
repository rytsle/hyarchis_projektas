from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocess import df
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

X = vectorizer.fit_transform(df['text'])

# Encode sentiment labels with LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(df['sentiment'])
print(f"Classes: {label_encoder.classes_}")
print(f"Encoded labels: {label_encoder.transform(['negative', 'neutral', 'positive'])}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf')
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

# f1 : negative : 0.65, positive : 0.70, neutral : 0.76, avg : 0.70
# precision : negative : 0.73, positive : 0.64, neutral : 0.81, avg : 0.73