import torch
from .base import TrainerBase
from ..callback import ProgressBar
from ..utils.tensor import tensor_to_cpu


class TextClassifierTrainer(TrainerBase):
    '''
    文本分类
    '''
    def _predict_forward(self, model, data_loader, do_eval, **kwargs):
        self.build_record_object()
        pbar = ProgressBar(n_total=len(data_loader), desc='Evaluating' if do_eval else 'Predicting')
        for step, batch in enumerate(data_loader):
            model.eval()
            inputs = self.build_inputs(batch)
            with torch.no_grad():
                outputs = model(**inputs)
            if do_eval:
                loss, logits = outputs[:2]
                loss = loss.mean()
                labels = inputs['labels']
                self.records['target'].append(tensor_to_cpu(labels))
                self.records['loss_meter'].update(loss.item(), n=1)
            else:
                if outputs[0].dim() == 1 and outputs[0].size(0) == 1:
                    logits = outputs[1]
                else:
                    logits = outputs[0]
            self.records['preds'].append(tensor_to_cpu(logits))
            pbar(step)
        self.records['preds'] = torch.cat(self.records['preds'], dim=0)
        if do_eval:
            self.records['target'] = torch.cat(self.records['target'], dim=0)
