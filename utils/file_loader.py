def load_file(file_path):
    ext = file_path.split('.')[-1]
    if ext == 'csv':
        return pd.read_csv(file_path)
    elif ext in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif ext == 'txt':
        with open(file_path, 'r') as f:
            return f.read()
    elif ext == 'pdf':
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif ext == 'docx':
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in ['png', 'jpg', 'jpeg']:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    else:
        raise ValueError("Unsupported file type")
