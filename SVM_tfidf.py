from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocess import df

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['text'])
Y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model and vectorizer for future use
import joblib   
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# f1 : negative : 0.65, positive : 0.70, neutral : 0.76, avg : 0.70
# precision : negative : 0.73, positive : 0.64, neutral : 0.81, avg : 0.73