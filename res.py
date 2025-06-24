import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to extract text from a single PDF file
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

# Load resumes from local folder
def load_resumes_from_folder(folder_path):
    pdf_texts = []
    resume_names = []

    print("\n📄 Extracted Resume Texts:\n")
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(path)
            pdf_texts.append(text)
            resume_names.append(filename)
            print(f"--- {filename} ---\n{text[:1000]}\n{'='*50}\n")
    return pdf_texts, resume_names

# Compute cosine similarity
def compute_similarity(job_description, resumes, names):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    job_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resumes)
    scores = cosine_similarity(job_embedding, resume_embeddings)[0]
    scored_resumes = list(zip(names, resumes, scores))
    scored_resumes.sort(key=lambda x: x[2], reverse=True)
    return scored_resumes

if __name__ == "__main__":
    # === 1. Load resumes ===
    folder_path = "resumes"  # Put all your PDF resumes inside this folder
    pdf_texts, resume_names = load_resumes_from_folder(folder_path)

    # === 2. Job description ===
    job_description = """
    We are looking for a Python Developer with experience in Flask, REST APIs,
    and working knowledge of databases like PostgreSQL. Familiarity with Docker and Git is a plus.
    """

    # === 3. Rank resumes ===
    results = compute_similarity(job_description, pdf_texts, resume_names)

    print("\n📊 Resume Ranking by Similarity:\n")
    for i, (name, _, score) in enumerate(results, 1):
        print(f"{i}. {name} — Similarity Score: {score:.2f}")

    # === 4. Evaluation ===
    num_resumes = len(results)
    print(f"\n📦 Total Resumes Processed: {num_resumes}")

    # Update true labels as per your known resume relevance (1 = relevant, 0 = not relevant)
    true_labels = [1] * num_resumes  # ← Change this based on actual relevance

    if len(true_labels) != num_resumes:
        raise ValueError("Mismatch between true labels and uploaded resumes!")

    threshold = 0.6
    similarity_scores = [score for _, _, score in results]
    predicted_labels = [1 if score >= threshold else 0 for score in similarity_scores]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print("\n📈 Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
