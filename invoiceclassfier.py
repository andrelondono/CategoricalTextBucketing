import PyPDF2
import textract
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Install necessary packages if not already installed
# pip install PyPDF2 textract nltk scikit-learn

# Download NLTK data
nltk.download('punkt')

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

# Step 2: Preprocess text data
def preprocess_text(text):
    # You may need more advanced text preprocessing based on your specific case
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

# Step 3: Train a simple text classification model
def train_text_classifier(X, y):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    return classifier, vectorizer

# Step 4: Save trained model and vectorizer
def save_model_and_vectorizer(classifier, vectorizer, model_path, vectorizer_path):
    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Step 5: Load model and vectorizer for inference
def load_model_and_vectorizer(model_path, vectorizer_path):
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return classifier, vectorizer

# Step 6: Use the trained model to predict the category of a new invoice
def predict_invoice_category(invoice_text, classifier, vectorizer):
    preprocessed_text = preprocess_text(invoice_text)
    features = vectorizer.transform([preprocessed_text])
    category = classifier.predict(features)[0]
    return category

# Example usage
if __name__ == "__main__":
    # Step 1: Extract text from PDF
    pdf_path = 'path/to/your/invoice.pdf'
    invoice_text = extract_text_from_pdf(pdf_path)

    # Step 2: Preprocess text data
    preprocessed_text = preprocess_text(invoice_text)

    # Step 3: Train or load a text classification model
    # For training:
    # labels = ['category1', 'category2', ...]
    # X = ['text1', 'text2', ...]
    # classifier, vectorizer = train_text_classifier(X, labels)
    # Save the trained model and vectorizer
    # save_model_and_vectorizer(classifier, vectorizer, 'model.pkl', 'vectorizer.pkl')

    # For loading pre-trained model and vectorizer
    classifier, vectorizer = load_model_and_vectorizer('model.pkl', 'vectorizer.pkl')

    # Step 4: Predict the category of the invoice
    predicted_category = predict_invoice_category(invoice_text, classifier, vectorizer)

    print(f'Predicted Category: {predicted_category}')
