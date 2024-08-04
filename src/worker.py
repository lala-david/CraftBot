import json
import time
from tqdm import tqdm
from langchain_ollama import ChatOllama
from embeddings import calculate_category_embeddings, get_embedding, find_most_similar_category

final3_path = 'xxxx'
label_info_path = 'xxxx'

with open(final3_path, 'r') as file:
    final3_data = json.load(file)

with open(label_info_path, 'r') as file:
    label_info_data = json.load(file)

categories = ["Wallet", "Proxy", "Exchange", "Swap", "Null Address", "Bridge", "Mixer", "Gambling", "Bot", "Deployer", "Other", "Exploiter"]

llm = ChatOllama(model="llama3.1:70b")

def get_category_label(description, llm):
    response = llm.chat(
        [
            {"role": "system", "content": "I'm a labeling bot."},
            {"role": "user", "content": f"Categorize the following label: {description}"}
        ],
        max_tokens=10
    )
    label = response['choices'][0]['message']['content'].strip()
    return label

category_examples = {category: [] for category in categories}
for event in final3_data.values():
    for category, labels in event.items():
        if category in category_examples:
            category_examples[category].extend(labels)

all_category_labels = []
for examples in category_examples.values():
    all_category_labels.extend(examples)

start_time = time.time()

print("임베딩중...")
category_embeddings = calculate_category_embeddings(all_category_labels)

labeled_data = {}

print("라벨중...")
for event_id, labels in tqdm(label_info_data.items(), desc="Processing events"):
    categorized_labels = {}

    for label in tqdm(labels, desc=f"Processing labels for event {event_id}", leave=False):
        label_embedding = get_embedding(label)
        matched_label = find_most_similar_category(label_embedding, category_embeddings, all_category_labels)

        category = None
        for cat, examples in category_examples.items():
            if matched_label in examples:
                category = cat
                break

        if category is None:
            category = "New"

        if category not in categorized_labels:
            categorized_labels[category] = []
        categorized_labels[category].append(label)

    labeled_data[event_id] = categorized_labels

output_path = 'labeled_results.json'
print(f"라벨링 저장됨요 {output_path}...")
with open(output_path, 'w') as output_file:
    json.dump(labeled_data, output_file, indent=4)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"총 소요시간: {elapsed_time:.2f} 초.")