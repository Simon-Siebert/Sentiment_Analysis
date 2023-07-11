import re
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.data.path.append('path_to_nltk_data')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Text preprocessing
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text

# Load the dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv')

# Rename the columns
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

# Map sentiment labels to human-readable values
label_sentiment = {0: "Negative", 2: "Neutral", 4: "Positive"}
df.sentiment = df.sentiment.map(label_sentiment)

# Text preprocessing on the 'text' column
df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Perform exploratory data analysis (EDA)
sns.countplot(x='sentiment', data=df)
plt.show()
plt.clf()

# Split the dataset into training and testing sets
X = df['preprocessed_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bag-of-Words Encoding
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train and evaluate the Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_bow, y_train)
y_pred = naive_bayes.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Plotting the accuracy scores
epochs = range(1, 10) # List of epochs (you can customize this)
accuracies = []  # List to store accuracy scores

for epoch in epochs:
    naive_bayes.fit(X_train_bow, y_train)
    y_pred = naive_bayes.predict(X_test_bow)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores of Naive Bayes Classifier')
plt.show()