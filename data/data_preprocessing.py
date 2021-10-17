from typing import List, Tuple
from pandas.core.frame import DataFrame

def raw_lines_to_label(lines: List[str]) -> List[str]:
    labels = []
    for line in lines:
        label = line.split(":")[0]
        labels.append(label)
    return labels

def get_labels_from(fileName: str) -> List[str]:
    labels = []
    with open(fileName) as raw:
        labels = raw_lines_to_label(raw)
    return labels

def remove_leading_trailing_space_from(dataset: DataFrame) -> DataFrame:
    return dataset.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def split_dataframe_into_X_y(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    y = df.iloc[: , -1:]
    X = df.iloc[:, :-1]

    return X, y
    
def map_classes(df: DataFrame, mapper: dict) -> DataFrame:
    return df.iloc[: , -1:].replace(mapper)