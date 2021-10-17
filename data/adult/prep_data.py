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

print(adult.describe)
print(adult.dtypes)

adult = adult.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# set random seed for reproducibility
np.random.seed(0)

print(adult.head())

# get the number of missing values
missing_values_count = adult.isnull().sum()

print(missing_values_count)

# get ? values
question_values = adult.isin(["?"]).sum()

print(question_values)