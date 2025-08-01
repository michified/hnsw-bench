with open("fashion-mnist_train.txt", "r") as f:
    lines = f.readlines()
    
data = []
for line in lines:
    parts = line.strip().split(",")
    label = int(parts[0])
    pixels = parts[1:]
    data.append((label, pixels))

with open("dataset.txt", "w") as f2:
    for elem in data:
        f2.write(' '.join(elem[1]) + '\n')

with open("fashion-mnist_test.txt", "r") as f3:
    lines = f3.readlines()
    
data = []
for line in lines:
    parts = line.strip().split(",")
    label = int(parts[0])
    pixels = parts[1:]
    data.append((label, pixels))

with open("queries.txt", "w") as f4:
    for elem in data:
        f4.write(' '.join(elem[1]) + '\n')
