import pandas as pd
from typing import List
import numpy as np

def raw_lines_to_label(lines: List[str]) -> List[str]:
    labels = []
    for line in lines:
        label = line.split(":")[0]
        labels.append(label)
    return labels

def get_labels_from_raw(fileName: str) -> List[str]:
    labels = []
    with open(fileName) as raw:
        labels = raw_lines_to_label(raw)
    return labels

adult = pd.read_csv("adult.data")

adult.columns = get_labels_from_raw("adult.columns_raw") 

# set random seed for reproducibility
np.random.seed(0)

print(adult.head())

