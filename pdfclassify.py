import fitz
from transformers import CLIPProcessor, CLIPModel
import torch

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    doc.close()
    return text

def identify_text(input_text, model, processor, class_labels):
    inputs = processor(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    label = class_labels[torch.argmax(probs)]  # the label with the highest probability is our prediction!
    return label

# Replace 'openai/clip-vit-base-patch4' with the desired CLIP model
model_name = 'openai/clip-vit-base-patch4'
pdf_path = 'your_pdf_path.pdf'

# Load CLIP model and processor
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define class labels (adjust according to your use case)
class_labels = ['unstructured_text', 'structured_text']

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Identify and classify text
prediction = identify_text(pdf_text, model, processor, class_labels)

# Print the result
print(f"The identified text in the PDF is: {prediction}")
