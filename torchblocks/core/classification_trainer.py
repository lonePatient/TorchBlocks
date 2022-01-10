import torch
from torchblocks.core import TrainerBase


class TextClassifierTrainer(TrainerBase):
    '''
    文本分类
    '''

    def predict_forward(self, batch):
        self.model.eval()
        inputs = self.build_batch_inputs(batch)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if 'loss' in outputs:
            outputs['loss'] = outputs['loss'].mean().detach().item()
        outputs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in
                   outputs.items()}
        batch = {key: value for key, value in dict(batch, **outputs).items() if
                 key not in self.keys_to_ignore_on_result_save}
        return batch
