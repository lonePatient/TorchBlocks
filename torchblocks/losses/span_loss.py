import torch.nn as nn
from torch.nn.modules.loss import _Loss


class SpanLoss(_Loss):
    def __init__(self, alpha=1.0, ignore_index=-1, name='Span Cross Entropy Loss'):
        super().__init__()
        self.alpha = alpha
        self.name = name
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input, target, masks=None):
        # assert if inp and target has both start and end values
        assert len(input) == 2, "start and end logits should be present for span loss calc"
        assert len(target) == 2, "start and end logits should be present for span loss calc"
        active_loss = masks.view(-1) == 1
        start_logits, end_logits = input
        start_positions, end_positions = target

        start_logits = start_logits.view(-1, start_logits.size(-1))
        end_logits = end_logits.view(-1, start_logits.size(-1))

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_positions.view(-1)[active_loss]
        active_end_labels = end_positions.view(-1)[active_loss]

        start_loss = self.loss_fct(active_start_logits, active_start_labels)
        end_loss = self.loss_fct(active_end_logits, active_end_labels)
        total_loss = (start_loss + end_loss) / 2
        return total_loss
