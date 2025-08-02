import numpy as np
import hnswlib
import time

k = 100

x_train = []
with open("strategies\\data\\dataset.txt", "r") as f:
    for line in f:
        vector = np.array([int(x) for x in line.strip().split()], dtype=np.float32) / 255.0
        x_train.append(vector)
x_train = np.array(x_train)

num_vectors, dimension = x_train.shape

build_start = time.time()

index = hnswlib.Index(space='l2', dim=dimension)

# M=16 is default, ef_construction controls build time vs accuracy tradeoff
index.init_index(max_elements=num_vectors, ef_construction=100, M=16)

# Add items
index.add_items(x_train)

index.save_index('strategies\\fashion-mnist.hnsw')

build_end = time.time()
build_time = build_end - build_start

index = hnswlib.Index(space='l2', dim=dimension)
index.load_index('strategies\\fashion-mnist.hnsw', max_elements=num_vectors)

index.set_ef(50)

queries = []
with open("strategies\\data\\queries.txt", "r") as f:
    for line in f:
        vector = np.array([int(x) for x in line.strip().split()], dtype=np.float32) / 255.0
        queries.append(vector)
queries = np.array(queries)

start_time = time.time()
results = []
for query in queries:
    labels, _ = index.knn_query(query, k=k)
    results.append(labels[0])
end_time = time.time()

topk = []
with open("strategies\\data\\topk.txt", "r") as f:
    for line in f:
        topk.append([int(x) for x in line.strip().split()])

total_recall = 0
for i in range(len(queries)):
    correct_neighbors = set(topk[i][:k])
    retrieved_neighbors = set(results[i])
    intersection_size = len(correct_neighbors.intersection(retrieved_neighbors))
    recall = intersection_size / k
    total_recall += recall

average_recall = total_recall / len(queries)

average_time_per_query = (end_time - start_time) / len(queries) * 1000000
queries_per_second = len(queries) / (end_time - start_time)

print(f"HNSW build time: {build_time:.4f} seconds")
print(f"Average time per query: {average_time_per_query:.4f} microseconds")
print(f"Queries per second: {queries_per_second:.4f}")
print(f"Average recall@{k}: {average_recall * 100:.4f}%")
