import torch
from torchblocks.callback import ProgressBar
from torchblocks.trainer.base import BaseTrainer
from torchblocks.utils.tensor import tensor_to_cpu
from torchblocks.losses.triplet_loss import DISTANCE2METRIC


class TripleTrainer(BaseTrainer):
    '''
    triple 分类
    '''
    def predict_step(self, model, data_loader, do_eval, **kwargs):
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
                self.records['target'].append(tensor_to_cpu(inputs['labels']))
                self.records['loss_meter'].update(loss.item(), n=1)
            else:
                if outputs[0].dim() == 1 and outputs[0].size(0) == 1:
                    logits = outputs[1]
                else:
                    logits = outputs[0]
            anchor, positive, negative = logits
            distance_metric = DISTANCE2METRIC[self.args.distance_metric]
            distance_positive = distance_metric(anchor, positive)
            distance_negative = distance_metric(anchor, negative)
            diff_dist = 1 - (distance_positive > distance_negative).int()
            self.records['preds'].append(tensor_to_cpu(diff_dist))
            pbar(step)
        self.records['preds'] = torch.cat(self.records['preds'], dim=0)
        if do_eval:
            self.records['target'] = torch.cat(self.records['target'], dim=0)
