import json
import time
from tqdm import tqdm
from embeddings import cal_cat_embedding, get_embedding, find_most_similar_category

prc_dt = 'xxxx'
raw_dt = 'xxxx'
lad_dt = {}

with open(prc_dt, 'r') as file:
    prc = json.load(file)

with open(raw_dt, 'r') as file:
    raw = json.load(file)

categories = [""]

category_emp = {category: [] for category in categories}
for event in prc.values():
    for category, labels in event.items():
        if category in category_emp:
            category_emp[category].extend(labels)

acl = []
for emp in category_emp.values():
    acl.extend(emp)

start_time = time.time()

print("임베딩중...")
cat_embedding = cal_cat_embedding(acl)

print("라벨중...")
for event_id, labels in tqdm(raw.items(), desc="전체 라벨링"):
    ct_label = {}

    for label in tqdm(labels, desc=f"부분 별 라벨링 {event_id}", leave=False):
        la_em = get_embedding(label)
        matched_label = find_most_similar_category(la_em, cat_embedding, acl)

        category = None
        for cat, emp in category_emp.items():
            if matched_label in emp:
                category = cat
                break

        if category is None:
            category = "New"

        if category not in ct_label:
            ct_label[category] = []
        ct_label[category].append(label)

    lad_dt[event_id] = ct_label

output_path = 'labeled_results.json'
print(f"라벨링 저장됨요 {output_path}...")
with open(output_path, 'w') as output_file:
    json.dump(lad_dt, output_file, indent=4)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"총 소요시간: {elapsed_time:.2f} 초.")
