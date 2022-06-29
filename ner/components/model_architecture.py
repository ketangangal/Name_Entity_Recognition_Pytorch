import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from ner.exception.exception import CustomException
import sys
import logging

logger = logging.getLogger(__name__)


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        """
        Config : Contains configuration for XLM roberta model type dict
        needs to be passed from user after configuration over-ridding.
        """
        super().__init__(config)
        logger.info("Model Initiated")
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
        try:
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
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)
