import numpy as np
labels = np.array(model.test_labels)
print(np.unique(labels, return_counts=True))
"""
Script to check the label distribution in your IMDB dataset splits
"""
import numpy as np
from datasets import load_dataset

# Load IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# Get all labels
train_labels = np.array(dataset["train"]["label"])
test_labels = np.array(dataset["test"]["label"])

print("\n" + "="*50)
print("ORIGINAL IMDB DATASET")
print("="*50)
print(f"Train set size: {len(train_labels)}")
print(f"Train label distribution: {np.bincount(train_labels)}")
print(f"  - Negative (0): {np.sum(train_labels == 0)}")
print(f"  - Positive (1): {np.sum(train_labels == 1)}")

print(f"\nTest set size: {len(test_labels)}")
print(f"Test label distribution: {np.bincount(test_labels)}")
print(f"  - Negative (0): {np.sum(test_labels == 0)}")
print(f"  - Positive (1): {np.sum(test_labels == 1)}")

# Simulate your 70/15/15 split
from sklearn.model_selection import train_test_split

print("\n" + "="*50)
print("YOUR 70/15/15 SPLIT (what it should be)")
print("="*50)

# Combine train and test from IMDB
all_labels = np.concatenate([train_labels, test_labels])
all_indices = np.arange(len(all_labels))

# Split: 70% train, 30% temp
train_idx, temp_idx = train_test_split(
    all_indices, 
    test_size=0.3, 
    random_state=42,
    stratify=all_labels
)

# Split temp into 15% val, 15% test
temp_labels = all_labels[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

train_split_labels = all_labels[train_idx]
val_split_labels = all_labels[val_idx]
test_split_labels = all_labels[test_idx]

print(f"Train split size: {len(train_split_labels)}")
print(f"Train label distribution: {np.bincount(train_split_labels)}")
print(f"  - Negative (0): {np.sum(train_split_labels == 0)}")
print(f"  - Positive (1): {np.sum(train_split_labels == 1)}")

print(f"\nValidation split size: {len(val_split_labels)}")
print(f"Val label distribution: {np.bincount(val_split_labels)}")
print(f"  - Negative (0): {np.sum(val_split_labels == 0)}")
print(f"  - Positive (1): {np.sum(val_split_labels == 1)}")

print(f"\nTest split size: {len(test_split_labels)}")
print(f"Test label distribution: {np.bincount(test_split_labels)}")
print(f"  - Negative (0): {np.sum(test_split_labels == 0)}")
print(f"  - Positive (1): {np.sum(test_split_labels == 1)}")

print("\n" + "="*50)
print("DIAGNOSIS")
print("="*50)
print("If your test set only shows negative examples,")
print("you likely have a bug in your data splitting code!")
print("\nEach split should have roughly equal positive/negative examples.")