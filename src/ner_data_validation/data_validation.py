import pandas as pd


class Validate:
    def __init__(self, data):
        self.train_data = data["train"]
        self.test_data = data["test"]
        self.validation_data = data["validation"]

    def check_columns_names(self):
        result_train = pd.DataFrame(self.train_data).columns == ['tokens', 'ner_tags', 'langs']
        result_test = pd.DataFrame(self.train_data).columns == ['tokens', 'ner_tags', 'langs']
        result_validate = pd.DataFrame(self.train_data).columns == ['tokens', 'ner_tags', 'langs']

        if sum(result_train) == sum(result_test) == sum(result_validate) == 3:
            return True
        else:
            return False

