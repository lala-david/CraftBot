from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformer('all-MiniLM-L12-v2')

def get_embedding(text):
    return embeddings.encode(text)

def calculate_category_embeddings(all_category_labels):
    return [get_embedding(label) for label in all_category_labels]

def find_most_similar_category(label_embedding, category_embeddings, all_category_labels):
    similarities = cosine_similarity([label_embedding], category_embeddings)
    category_index = similarities.argmax()
    return all_category_labels[category_index]