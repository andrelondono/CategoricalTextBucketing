import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

# Function to classify PDF files
def classify_pdf(pdf_folder):
    # Data preparation
    data = []
    labels = []

    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                data.append(text)
                labels.append(root)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create a text classification model using TF-IDF and Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    return model

# Example usage
pdf_folder_path = "/path/to/your/pdf/folder"
trained_model = classify_pdf(pdf_folder_path)

# Now you can use the trained_model to predict the category of new PDF files
# new_pdf_text = extract_text_from_pdf("/path/to/your/new/pdf/file.pdf")
# predicted_category = trained_model.predict([new_pdf_text])
# print(f"Predicted category: {predicted_category[0]}")
