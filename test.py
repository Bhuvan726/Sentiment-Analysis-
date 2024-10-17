import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk

# Download the NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
# Adjust the file path to your dataset location
dataframe = pd.read_csv(r"D:\Analysis\amazon_reviews.csv")

# Preview the dataset
print(dataframe.head())

# Data cleaning and preprocessing function
def clean_review_text(review):
    # Check if review is a string; if not, return an empty string
    if isinstance(review, str):
        # Tokenize the review text
        tokens = word_tokenize(review.lower())
        # Remove stopwords and non-alphanumeric tokens
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''  # Return an empty string for non-string inputs

# Apply preprocessing to the review text while handling NaN or float values
dataframe['cleaned_review'] = dataframe['reviewText'].apply(clean_review_text)

# Create labels (1 for positive, 0 for negative; adjust according to your dataset)
dataframe['label'] = np.where(dataframe['overall'] >= 4, 1, 0)  # Assuming 'overall' is the rating column

# Split the data into training and testing sets
features = dataframe['cleaned_review']
labels = dataframe['label']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
X_test_vectorized = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = logistic_model.predict(X_test_vectorized)

# Evaluate the model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))