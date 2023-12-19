#signature
import cv2
from PIL import Image
import numpy as np
import fitz 
from docx import Document
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from flask import Flask, request, jsonify


app = Flask(__name__)

def has_signature(pixmap):
    img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((pixmap.h, pixmap.w, 3))
    # pixmap.samples
    # np.array(pixmap.getImage())
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresholded_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

#!1signature pdf

def has_signature_in_pdf(pdf_path,Docum):
    doc = Docum
    # fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        if has_signature(pix):
            return True
    return False

#!2signature doc
def has_signature_in_docx(docx_path,Docum):
    doc = Docum
    # Document(docx_path)
    for rel in doc.part.rels:
        if "image" in str(doc.part.rels[rel].target_ref):
            image_data = doc.part.rels[rel].target_part.blob
            if has_signature(Image.open(io.BytesIO(image_data))):
                return True
    return False


def validateSign(document_path,Docum):
    if document_path.lower().endswith('.pdf'):
        return has_signature_in_pdf(document_path,Docum)
    elif document_path.lower().endswith('.docx'):
        return has_signature_in_docx(document_path,Docum)
    else:
        return False

#!3 check pg count and  min word count

nltk.download('punkt')
nltk.download('words')
def is_valid_document(document_path,Docum):
    min_word_count = 50
    if document_path.lower().endswith('.pdf'):
        doc = Docum
        # fitz.open(document_path)
        if doc.page_count < 1:
            return False
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()

            tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
            if len(tokens) >= min_word_count:
                return True
        return False
    elif document_path.lower().endswith('.docx'):
        from docx import Document

        doc = Document(document_path)
        if len(doc.paragraphs) < 1:
            return False  

        for paragraph in doc.paragraphs:
            text = paragraph.text

            tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]

            if len(tokens) >= min_word_count:
                return True

        return False

    else:
        return False


#!4 atlst 50


def contains_keywords(document_path, keywords,Docum):
    if document_path.lower().endswith('.pdf'):
        doc = Docum
        # fitz.open(document_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return True
        return False
    elif document_path.lower().endswith('.docx'):
        doc = Docum
        for paragraph in doc.paragraphs:
            text = paragraph.text
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return True
        return False
    else:
        return False
    
#! key word check


# if __name__=="__main__":
#     document_path = "path/to/your/uploaded/document.pdf" 
#     keywords_to_check = ["Loan", "Request", "Withdrawal", "Spousal", "Consent"]
#     keywrdResult = contains_keywords(document_path, keywords_to_check)
#     validateDocumentResult = is_valid_document(document_path)
#     signResult = validateSign(document_path)

#     jsonResult = {
#         "keywordsValid":keywrdResult,
#         "validateDoc": validateDocumentResult,
#         "signatureResult": signResult
#     }

#     print(jsonResult)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/validate_document', methods=['POST'])
def validate_document():
    # Get the uploaded file from the request
    print("21112")
    uploaded_file = request.files['document']
    keywords_to_check = ["Loan", "Request", "Withdrawal", "Spousal", "Consent"]

    # Read the content of the file
    content = uploaded_file.read()

    # Check if the document is a PDF
    if uploaded_file.filename.lower().endswith('.pdf'):
        doc = fitz.open("pdf", content)
        print(doc,"2111")
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            keywords_result = contains_keywords(text, keywords_to_check,doc)

            document_valid_result = is_valid_document(text,doc)

            signature_result = validateSign(uploaded_file.filename,doc)

            # Create JSON result
            json_result = {
                "keywords_valid": keywords_result,
                "document_valid": document_valid_result,
                "signature_result": signature_result
            }

            return jsonify(json_result)

if __name__ == '__main__':
    app.run(debug=True)
