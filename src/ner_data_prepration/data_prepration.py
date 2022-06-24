from transformers import AutoTokenizer
from transformers import AutoConfig


class Preprocessing:
    def __init__(self, en_data):
        self.en_data = en_data
        self.xlmr_model_name = "xlm-roberta-base"
        self.ner_tags = self.en_data["train"].features["ner_tags"].feature
        self.index2tag = {idx: tag for idx, tag in enumerate(self.ner_tags.names)}
        self.tag2index = {tag: idx for idx, tag in enumerate(self.ner_tags.names)}

    def create_tag_names(self, batch):
        return {"ner_tags_str": [self.ner_tags.int2str(idx) for idx in batch["ner_tags"]]}

    def tokenization(self):
        xlmr_tokenizer = AutoTokenizer.from_pretrained(self.xlmr_model_name)
        return xlmr_tokenizer

    def create_auto_config(self):
        xlmr_config = AutoConfig.from_pretrained(self.xlmr_model_name,
                                                 num_labels=self.ner_tags.num_classes,
                                                 id2label=self.index2tag,
                                                 label2id=self.tag2index)
        return xlmr_config

    def tokenize_and_align_labels(self, examples):
        xlmr_tokenizer = self.tokenization()
        tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True,
                                          is_split_into_words=True)
        labels = []
        for idx, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def encode_en_dataset(self, corpus):
        return corpus.map(self.tokenize_and_align_labels, batched=True,
                          remove_columns=['langs', 'ner_tags', 'tokens'])

    def prepare_data_for_fine_tuning(self):
        # Map Data with create_tag_names
        self.en_data = self.en_data.map(self.create_tag_names)

        # Map word-id and label id into main data
        panx_en_encoded = self.encode_en_dataset(self.en_data)
        return panx_en_encoded
