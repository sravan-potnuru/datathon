import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Split the data into training and testing sets
X = data['Text_of_Speech']
y = data['Emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=60)
                                                    

# Vectorize text data (TF-IDF vectorization)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)  # You can try different kernels

# Train the SVM model
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)
