import unittest

import sys

sys.path.append('./data/')
from data_preprocessing import *

import pandas as pd

class TestPreprocessingData(unittest.TestCase):

    def test_raw_lines_to_labels(self):
        expected_labels = ["A", "B"]
        self.assertEqual(get_labels_from("./data/test/test.names"), expected_labels)

    def test_remove_leading_trailing_space_from(self):
        space_df = pd.DataFrame({"a": ["test ", "  test"], "b": ["test  ", " test"]})
        expected_df = pd.DataFrame({"a": ["test", "test"], "b": ["test", "test"]})
        
        actual_df = remove_leading_trailing_space_from(space_df)
        
        self.assertTrue(expected_df.equals(actual_df))

    def test_split_dataframe_into_X_y(self):
        df = pd.DataFrame({"a": ["test", "test"], "b": ["test", "test"], "c": ["test", "test"],})
        expected_X = pd.DataFrame({"a": ["test", "test"], "b": ["test", "test"]})
        expected_y = pd.DataFrame({"c": ["test", "test"]})
        
        actual_X, actual_y = split_dataframe_into_X_y(df)

        self.assertTrue(expected_X.equals(actual_X))
        self.assertTrue(expected_y.equals(actual_y))

    def test_convert_classes(self):
        df = pd.DataFrame({"c": ["test0", "test1"],})
        expected_y = pd.DataFrame({"c": [0, 1]})

        mapper = {'test0': 0, 'test1': 1}
        actual_y = map_classes(df, mapper)
        
        self.assertTrue(expected_y.equals(actual_y))

if __name__ == '__main__':
    unittest.main()