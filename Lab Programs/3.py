import pandas as pd
from math import log2

# Load dataset
data = pd.read_csv("play_tennis.csv")

def entropy(column):
    values = column.value_counts()
    total = len(column)
    return sum([-(count/total) * log2(count/total) for count in values])

def info_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    values = data[attribute].value_counts()
    weighted_entropy = 0
    for v, count in values.items():
        subset = data[data[attribute] == v]
        weighted_entropy += (count/len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(data, target, attributes):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if not attributes:
        return data[target].mode()[0]

    gains = {attr: info_gain(data, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = data[target].mode()[0]
        else:
            remaining = [a for a in attributes if a != best_attr]
            tree[best_attr][value] = id3(subset, target, remaining)

    return tree

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
tree = id3(data, "PlayTennis", attributes)

print("Decision Tree:", tree)

# Classify a new sample
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0]
    value = sample[attribute]
    return classify(tree[attribute][value], sample)

sample = {
    "Outlook": "Sunny",
    "Temperature": "Mild",
    "Humidity": "High",
    "Wind": "Weak"
}

print("Classified Output:", classify(tree, sample))
