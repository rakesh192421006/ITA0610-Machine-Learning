import math

# ----- Sample training data -----
# Each item: ([feature1, feature2], class_label)
dataset = [
    ([1.0, 1.0], 0),
    ([1.5, 2.0], 0),
    ([2.0, 1.0], 0),
    ([6.0, 5.0], 1),
    ([7.0, 6.0], 1),
    ([8.0, 5.0], 1)
]

# ----- Euclidean distance -----
def euclidean_distance(x1, x2):
    s = 0.0
    for i in range(len(x1)):
        diff = x1[i] - x2[i]
        s += diff * diff
    return math.sqrt(s)

# ----- Get k nearest neighbours -----
def get_k_neighbours(dataset, test_point, k):
    distances = []
    for features, label in dataset:
        d = euclidean_distance(features, test_point)
        distances.append((d, label))
    # sort by distance
    distances.sort(key=lambda x: x[0])
    # take first k
    neighbours = distances[:k]
    return neighbours

# ----- Predict class by majority vote -----
def knn_predict(dataset, test_point, k):
    neighbours = get_k_neighbours(dataset, test_point, k)
    # count votes
    count0 = 0
    count1 = 0
    for d, label in neighbours:
        if label == 0:
            count0 += 1
        else:
            count1 += 1
    # choose class with max count
    if count0 > count1:
        return 0
    else:
        return 1

# ----- Main -----
if __name__ == "__main__":
    test_point = [2.0, 2.0]
    k = 3
    predicted_class = knn_predict(dataset, test_point, k)
    print("Test point:", test_point)
    print("Predicted class:", predicted_class)
