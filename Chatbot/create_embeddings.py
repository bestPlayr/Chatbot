# create_embeddings.py
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Step 1. Load PDF
# ----------------------------
pdf_path = "IncomeTaxOrdinance.pdf"
pdf_doc = fitz.open(pdf_path)

# ----------------------------
# Step 2. Extract normal text
# ----------------------------
all_text_docs = []

for page_num in range(27, len(pdf_doc) + 1):  # skip first 5 pages as TOC
    page = pdf_doc[page_num - 1]
    text = page.get_text("text").strip()
    if text:
        all_text_docs.append(
            Document(
                page_content=text,
                metadata={"source": "law_text", "page": page_num}
            )
        )

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = splitter.split_documents(all_text_docs)

# ----------------------------
# Step 3. Extract headings & table-like rows
# ----------------------------
def extract_heading(page_number: int) -> str:
    """Extract heading/title from a page (like 'First Schedule Part I')."""
    page = pdf_doc[page_number - 1]
    text = page.get_text("text")
    for line in text.split("\n"):
        if any(keyword in line for keyword in ["Schedule", "Part", "Section"]):
            return line.strip()
    return "Table"

def extract_table_like_content(page_number: int):
    """Extract structured rows from blocks (simulate tables)."""
    page = pdf_doc[page_number - 1]
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    rows = []
    for b in blocks:
        text = b[4].strip()
        if "|" in text or "\t" in text:  # crude heuristic for tabular content
            rows.append(text)
    return rows

table_docs = []
for page_num in range(1, len(pdf_doc) + 1):  # check all pages
    rows = extract_table_like_content(page_num)
    if not rows:
        continue
    heading = extract_heading(page_num)
    for i, row in enumerate(rows):
        table_docs.append(
            Document(
                page_content=row,
                metadata={"source": heading, "row": i + 1, "page": page_num}
            )
        )

# ----------------------------
# Step 4. Combine docs
# ----------------------------
all_docs = text_chunks + table_docs
print(f"✅ Loaded {len(text_chunks)} text chunks and {len(table_docs)} table-like rows.")

# ----------------------------
# Step 5. Store in ChromaDB
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="tax_law",
    persist_directory="./chroma_db"
)

print("✅ Embeddings created and stored in ./chroma_db")
