# ===============================================================
# MODEL EVALUATION: GPT-4o-mini vs LLaMA3-8B vs DistilGPT-2
# With Zero-Shot, Few-Shot, and RAG Evaluation (FAISS Index)
# ===============================================================

import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# -------- Load FAISS RAG Index --------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------- Model APIs --------
from openai import OpenAI
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------- Evaluation Metrics --------
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

# ===============================================================
# LOAD RAG INDEX (faiss_index_fast)
# ===============================================================

print("📌 Loading FAISS RAG index...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index_fast",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retrieval function
def retrieve_from_rag(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])


# ===============================================================
# PROMPTS
# ===============================================================

ZERO_SHOT_PROMPT = """
Explain what Reinforcement Learning is.
Mention practical applications and key challenges.
"""

FEW_SHOT_PROMPT = """
Q: What is supervised learning?
A: A method where a model learns from labeled examples.

Q: What is unsupervised learning?
A: A method where the model discovers patterns without labels.

Q: What is Reinforcement Learning?
A:
"""

# RAG prompt dynamically pulls context from FAISS
def build_rag_prompt(query):
    context = retrieve_from_rag(query)
    return f"""
Use the following retrieved context to answer the question:

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly and factually.
"""


QUERY = "Explain what Reinforcement Learning is and list its challenges."


# ===============================================================
# 1. LOAD MODELS
# ===============================================================

# ---------- GPT-4o-mini ----------
OPENAI_API_KEY =    getenv("OPENAI_API_KEY")
gpt_client = OpenAI(api_key=OPENAI_API_KEY)

def call_gpt(prompt):
    out = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350
    )
    return out.choices[0].message.content


# ---------- LLaMA3-8B ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def call_llama(prompt):
    out = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return out.choices[0].message.content


# ---------- DistilGPT-2 (local) ----------
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def call_distil(prompt):
    out = pipe(prompt, max_length=300, do_sample=True)[0]["generated_text"]
    return out[len(prompt):]


# ===============================================================
# 2. RUN ALL MODELS UNDER ALL SETTINGS
# ===============================================================

settings = {
    "Zero-Shot": ZERO_SHOT_PROMPT,
    "Few-Shot": FEW_SHOT_PROMPT,
    "RAG": build_rag_prompt(QUERY)
}

models = {
    "GPT-4o-mini": call_gpt,
    "LLaMA3-8B": call_llama,
    "DistilGPT-2": call_distil
}

outputs = {}

print("\n================ RUNNING EXPERIMENTS ================\n")

for setting_name, prompt in settings.items():
    print(f"\n▶ MODE: {setting_name}")
    outputs[setting_name] = {}

    for model_name, fn in models.items():
        print(f" → Running {model_name}...")
        outputs[setting_name][model_name] = fn(prompt)


# Save raw model outputs
os.makedirs("evaluation_outputs", exist_ok=True)
with open("evaluation_outputs/raw_outputs.json", "w") as f:
    json.dump(outputs, f, indent=4)


# ===============================================================
# 3. COMPUTE METRICS
# ===============================================================

bert_embedder = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

metrics = {}

for setting in outputs:
    metrics[setting] = {}

    reference = outputs[setting]["GPT-4o-mini"]  # baseline reference

    for model_name, text in outputs[setting].items():

        bleu = sentence_bleu([reference.split()], text.split())
        rouge_l = rouge.score(reference, text)['rougeL'].fmeasure
        _, _, F1 = bert_score([text], [reference], lang="en")

        e1 = bert_embedder.encode(reference, convert_to_tensor=True)
        e2 = bert_embedder.encode(text, convert_to_tensor=True)
        similarity = float(util.cos_sim(e1, e2)[0][0])

        metrics[setting][model_name] = {
            "BLEU": float(bleu),
            "ROUGE-L": float(rouge_l),
            "BERTScore": float(F1[0]),
            "SemanticSimilarity": similarity,
            "Length": len(text)
        }

# Save metrics
with open("evaluation_outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# ===============================================================
# 4. VISUALIZE RESULTS
# ===============================================================

for setting in metrics:
    for metric in ["BLEU", "ROUGE-L", "BERTScore", "SemanticSimilarity"]:

        plt.figure(figsize=(7, 4))
        plt.title(f"{metric} — {setting}")
        plt.bar(metrics[setting].keys(), [m[metric] for m in metrics[setting].values()])
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

        plt.savefig(f"evaluation_outputs/{setting}_{metric}.png")
        plt.close()

print("\n🎉 DONE! All results saved in evaluation_outputs/")
