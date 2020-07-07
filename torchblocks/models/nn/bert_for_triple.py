from torchblocks.losses import TripletLoss
from transformers import BertPreTrainedModel, BertModel


class BertForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTripletNet, self).__init__(config)
        self.bert = BertModel(config)
        self.distance_metric = config.distance_metric
        self.init_weights()

    def forward(self,
                a_input_ids,
                b_input_ids,
                c_input_ids,
                a_token_type_ids=None,
                b_token_type_ids=None,
                c_token_type_ids=None,
                a_attention_mask=None,
                b_attention_mask=None,
                c_attention_mask=None
                ):
        _, a_pooled_output = self.bert(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                       attention_mask=a_attention_mask)
        _, b_pooled_output = self.bert(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                       attention_mask=b_attention_mask)
        _, c_pooled_output = self.bert(input_ids=c_input_ids, token_type_ids=c_token_type_ids,
                                       attention_mask=c_attention_mask)
        outputs = ((a_pooled_output, b_pooled_output, c_pooled_output),)
        loss_fct = TripletLoss(distance_metric=self.distance_metric, average=True)
        loss = loss_fct(anchor=a_pooled_output, positive=b_pooled_output, negative=c_pooled_output)
        outputs = (loss,) + outputs
        return outputs
