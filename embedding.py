# ============================================================
#        FAST RAG INDEX BUILDER (64-turn chunks)
#         Alpaca + FAISS + MiniLM embeddings
# ============================================================

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
import gc

nltk.download("punkt")
nltk.download("punkt_tab")

# ============================================================
# STEP 1 — Load Alpaca
# ============================================================
print("Loading Alpaca dataset...")
dataset = load_dataset("tatsu-lab/alpaca")["train"]
print(f"Loaded {len(dataset)} examples.\n")


# ============================================================
# STEP 2 — Convert each example into a 'dialogue turn'
# ============================================================
def convert_to_turn(record):
    instruction = record["instruction"].strip()
    input_text = record["input"].strip()
    output = record["output"].strip()

    turn = f"Instruction: {instruction}\n"
    if input_text:
        turn += f"Input: {input_text}\n"
    turn += f"Output: {output}\n"

    return turn


dialogue_turns = [convert_to_turn(r) for r in dataset]


# ============================================================
# STEP 3 — Chunk into 64-turn overlapping windows
# ============================================================
def chunk_dialogue(turns, chunk_size=64, overlap=16):
    chunks = []
    i = 0

    while i < len(turns):
        window = turns[i : i + chunk_size]
        if not window:
            break

        chunk_text = "\n".join(window)
        chunks.append(Document(page_content=chunk_text))

        i += chunk_size - overlap  # slide window

    return chunks


print("Creating 64-turn chunks with 16-turn overlap...")
chunks = chunk_dialogue(dialogue_turns, chunk_size=64, overlap=16)
print(f"Total chunks created: {len(chunks)}\n")  # usually around 800–900


# ============================================================
# STEP 4 — Build FAISS Index
# ============================================================
print("Loading MiniLM embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS vector index...")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save
vectorstore.save_local("faiss_index_fast")
print("\n===========================================")
print("FAISS (FAST 64-turn) index saved to: faiss_index_fast/")
print("===========================================")
