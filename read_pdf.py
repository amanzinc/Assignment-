import sys
import io

# Set standard output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed")
    sys.exit(1)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    print(extract_text(sys.argv[1]))
