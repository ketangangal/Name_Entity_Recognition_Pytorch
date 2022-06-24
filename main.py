import pandas as pd
from datasets import load_dataset
import numpy as np 
import torch

from transformers import AutoTokenizer
from transformers import AutoConfig
from seqeval.metrics import classification_report
from datasets import get_dataset_config_names
from seqeval.metrics import f1_score
from transformers import DataCollatorForTokenClassification
from transformers import Trainer



xtreme_subsets = get_dataset_config_names("xtreme")
print(f"XTREME has {len(xtreme_subsets)} configurations")


panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")]


en = load_dataset("xtreme", name='PAN-X.en')

for i in en["train"]:
  print(i)
  break

pd.DataFrame(en["train"][1]).transpose()

for key, value in en["train"].features.items():
  print(f"{key}: {value}")

tags = en["train"].features["ner_tags"].feature
print(tags)

def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

new_en = en.map(create_tag_names)

de_example = new_en["train"][0]
pd.DataFrame([de_example["tokens"], de_example["ner_tags_str"]],
['Tokens', 'Tags'])

xlmr_model_name = "xlm-roberta-base"
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

xlmr_tokenizer('Hello my name is ketan and i am working in the ineuron')

xlmr_tokens = xlmr_tokenizer('Hello my name is ketan and i am working in the ineuron').tokens()
xlmr_tokens

import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, **kwargs)
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return TokenClassifierOutput(loss=loss, logits=logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, 
                                         num_labels=tags.num_classes,
                                         id2label=index2tag, label2id=tag2index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model = (XLMRobertaForTokenClassification
              .from_pretrained(xlmr_model_name, config=xlmr_config)
              .to(device))


text = 'Hello my name is ketan and i am working in the ineuron'
input_ids = xlmr_tokenizer.encode(text, return_tensors="pt")

outputs = xlmr_model(input_ids.to(device)).logits
predictions = torch.argmax(outputs, dim=-1)
print(f"Number of tokens in sequence: {len(xlmr_tokens)}")
print(f"Shape of outputs: {outputs.shape}")


preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
pd.DataFrame([xlmr_tokens, preds], index=["Tokens", "Tags"])


def tag_text(text, tags, model, tokenizer):
    # Get tokens with special characters
    tokens = tokenizer(text).tokens()
    # Encode the sequence into IDs
    input_ids = xlmr_tokenizer(text, return_tensors="pt").input_ids.to(device)
    # Get predictions as distribution over 7 possible classes
    outputs = model(input_ids)[0]
    # Take argmax to get most likely class per token
    predictions = torch.argmax(outputs, dim=2)
    # Convert to DataFrame
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["Tokens", "Tags"])



words, labels = de_example["tokens"], de_example["ner_tags"]
print(words, labels)

tokenized_input = xlmr_tokenizer(de_example["tokens"], is_split_into_words=True)
print(tokenized_input)

tokens = xlmr_tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)

word_ids = tokenized_input.word_ids()
print(word_ids)

previous_word_idx = None
label_ids = []

for word_idx in word_ids:
    if word_idx is None or word_idx == previous_word_idx:
        label_ids.append(-100)
    elif word_idx != previous_word_idx:
        label_ids.append(labels[word_idx])
    previous_word_idx = word_idx

labels = [index2tag[l] if l != -100 else "IGN" for l in label_ids]
index = ["Tokens", "Word IDs", "Label IDs", "Labels"]

pd.DataFrame([tokens, word_ids, label_ids, labels], index=index)


def tokenize_and_align_labels(examples):
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

def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True,
                      remove_columns=['langs', 'ner_tags', 'tokens'])

panx_en_encoded = encode_panx_dataset(new_en) 

## Prediction Matrics

import numpy as np

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list

from transformers import TrainingArguments

num_epochs = 1

batch_size = 24

logging_steps = len(panx_en_encoded["train"]) // batch_size

model_name = f"{xlmr_model_name}-finetuned-panx-en"

training_args = TrainingArguments(
    output_dir=model_name, 
    log_level="error", 
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size, 
    evaluation_strategy="epoch",
    save_steps=1e6, 
    weight_decay=0.01,
    disable_tqdm=False,
    logging_steps=logging_steps
    )
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

def model_init():
    return (XLMRobertaForTokenClassification
            .from_pretrained(xlmr_model_name, config=xlmr_config)
            .to(device))

trainer = Trainer(model_init=model_init, args=training_args,
                  data_collator=data_collator, compute_metrics=compute_metrics,
                  train_dataset=panx_en_encoded["train"],
                  eval_dataset=panx_en_encoded["validation"],
                  tokenizer=xlmr_tokenizer)


# Training

trainer.train() 

text_de = "ketan Gangal is working at iNeuron in Nlp Segment"
tag_text(text_de, tags, trainer.model, xlmr_tokenizer)

# Model Prediction

input_ids = xlmr_tokenizer(new_en["validation"][:10]["tokens"],truncation=True,
                                      is_split_into_words=True)

data = torch.tensor(panx_en_encoded["validation"][0:1]["input_ids"])


outputs = xlmr_model(data.to(device)).logits
predictions = torch.argmax(outputs, dim=-1)
print(f"Number of tokens in sequence: {len(xlmr_tokens)}")
print(f"Shape of outputs: {outputs.shape}")

tokens = xlmr_tokenizer.convert_ids_to_tokens(data[0])
print(" ".join(tokens))

pred_tags = [index2tag[i.item()] for i in predictions[0]][1:-1]
print(pred_tags)

" ".join([index2tag[i.item()] for i in predictions[0]])


output = xlmr_tokenizer.convert_tokens_to_string(tokens)

pd.Series([index2tag[i] for i in [i for i in panx_en_encoded["validation"][0]['labels'] if i!= -100 ]])

panx_en_encoded["validation"][0]['labels']

xlmr_tokenizer.convert_ids_to_tokens(panx_en_encoded["validation"][0]["input_ids"])

panx_en_encoded["validation"][0]['ner_tags_str']