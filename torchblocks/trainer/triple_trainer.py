import torch
from .base import TrainerBase
from ..callback import ProgressBar
from ..utils.tensor import tensor_to_cpu
from ..losses.triplet_loss import DISTANCE2METRIC


class TripleTrainer(TrainerBase):
    def __init__(self, args, metrics, logger, batch_input_keys, collate_fn=None):

        super().__init__(args=args,
                         metrics=metrics,
                         logger=logger,
                         batch_input_keys=batch_input_keys,
                         collate_fn=collate_fn)

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
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                labels = inputs['labels']
                self.records['target'].append(tensor_to_cpu(labels))
                self.records['loss_meter'].update(loss.item(), n=1)
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
