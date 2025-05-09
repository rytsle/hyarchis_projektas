import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
import string

# Make sure NLTK resources are available
try:
    nltk.download('punkt')
except:
    print("Failed to download automatically, please run the following commands manually:")
    print("import nltk")
    print("nltk.download('punkt')")

# Load data from preprocess.py
from preprocess import df

# Additional preprocessing to improve text quality
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply additional cleaning
df['text_cleaned'] = df['text'].apply(clean_text)

# Tokenize the text
df['preprocessed_text'] = df['text_cleaned'].apply(lambda x: word_tokenize(x.lower()))

# Check class distribution
class_counts = df['sentiment'].value_counts()
print("Class distribution:")
print(class_counts)

# Train Word2Vec model with optimized but faster parameters
sentences = df['preprocessed_text'].tolist()
word2vec_model = Word2Vec(
    sentences=sentences, 
    vector_size=150,     # Moderate dimension size
    window=6,            # Moderate context window
    min_count=2,         # Ignore rare words to reduce noise
    workers=4,           # Use multiple cores
    sg=0,                # CBOW is faster than skip-gram
    epochs=10            # Fewer epochs for faster training
)
print(f"Word2Vec model trained with vocabulary size: {len(word2vec_model.wv.key_to_index)}")

# Create document vectors with simple IDF weighting (faster than full TF-IDF)
from collections import Counter, defaultdict

# Calculate document frequency for each word
word_counts = Counter()
for text in sentences:
    # Count each unique word only once per document
    unique_words = set(text)
    for word in unique_words:
        word_counts[word] += 1

# Calculate simplified IDF
num_docs = len(sentences)
word_idf = defaultdict(lambda: 1.0)
for word, count in word_counts.items():
    word_idf[word] = np.log(num_docs / (1 + count))

# Create document vectors using IDF weighted average (faster than full TF-IDF)
doc_vectors = []
for text in tqdm(sentences, desc="Creating document vectors"):
    word_vectors = []
    weights = []
    
    for word in text:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])
            weights.append(word_idf[word])  # Use the word's IDF as weight
    
    if len(word_vectors) > 0:
        # Weighted average of word vectors
        weighted_vectors = np.array(word_vectors) * np.array(weights).reshape(-1, 1)
        doc_vector = weighted_vectors.sum(axis=0) / np.sum(weights)
    else:
        doc_vector = np.zeros(word2vec_model.vector_size)
        
    doc_vectors.append(doc_vector)
X = np.array(doc_vectors)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode sentiment labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])
print(f"Classes: {label_encoder.classes_}")

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM with linear kernel (much faster than GridSearchCV)
classifier = SVC(
    C=10,               # Higher C value for potentially better accuracy
    kernel='linear',    # Linear kernel is usually faster for text data
    probability=True,
    class_weight=class_weight_dict,
    random_state=42
)

print("Training SVM classifier...")
classifier.fit(X_train, y_train)
print("SVM model trained")

# Evaluate classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Process completed successfully.")