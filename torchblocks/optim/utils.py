def get_optimizer_params(model, lr, lr_weight_decay_coef, num_layers):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if lr_weight_decay_coef < 1.0:
        optimizer_grouped_parameters = [
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                   and 'bert.encoder' not in n
                   and not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                   and 'bert.encoder' not in n
                   and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                   and not any(nd in n for nd in no_decay)],
                'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                   and any(nd in n for nd in no_decay)],
                'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.0}
        ]
        for i in range(num_layers):
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                       and any(nd in n for nd in no_decay)],
                    'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                       and any(nd in n for nd in no_decay)],
                    'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters
