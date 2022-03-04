from torchblocks.core import TrainerBase


class SequenceLabelingTrainer(TrainerBase):

    def build_batch_concat(self, all_batch_list, dim=0):
        preds = []
        target = []
        for batch in all_batch_list:
            preds.extend(batch['predictions'])
            target.extend(batch['groundtruths'])
        return {"preds":preds, "target":target}


