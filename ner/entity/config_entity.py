from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_name", "subset_name", "data_path"])

DataValidationConfig = namedtuple("DataValidationConfig", ["dataset", "data_split", "columns_check",
                                                           "type_check", "null_check"])

DataPreprocessingConfig = namedtuple("DataPreprocessingConfig", ["model_name", "tags", "index2tag",
                                                                 "tag2index", "tokenizer"])

ModelTrainConfig = namedtuple("ModelTrainConfig", ["model_name", "index2tag", "tag2index",
                                                   "tokenizer", "xlmr_config", "epochs",
                                                   "batch_size", "save_steps", "output_dir"])

PredictPipelineConfig = namedtuple("PredictPipelineConfig", ["tokenizer", "truncation", "is_split_into_words",
                                                             "output_dir", "index2tag", "tag2index"])

TestPipelineConfig = namedtuple("TestPipelineConfig", ["dataset_name", "subset_name"])
