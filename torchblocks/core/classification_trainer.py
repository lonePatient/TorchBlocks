import torch
from torchblocks.core import TrainerBase


class TextClassifierTrainer(TrainerBase):
    '''
    文本分类
    '''
    def build_batch_concat(self, all_batch_list, dim=0):
        preds = torch.cat([batch['logits'] for batch in all_batch_list], dim=dim)
        target = torch.cat([batch['labels'] for batch in all_batch_list], dim=dim)
        return {"preds": preds, "target": target}

