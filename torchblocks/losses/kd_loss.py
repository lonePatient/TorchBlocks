import torch.nn.functional as F
import torch
import torch.nn as nn


class CosLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_S, state_T, mask=None):
        '''
        This is the loss used in DistilBERT

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            state_S = state_S.view(-1, state_S.size(-1))
            state_T = state_T.view(-1, state_T.size(-1))
        else:
            mask = mask.to(state_S).unsqueeze(-1).expand_as(state_S).to(torch.uint8)  # (bs,len,dim)
            state_S = torch.masked_select(state_S, mask).view(-1, mask.size(-1))  # (bs * select, dim)
            state_T = torch.masked_select(state_T, mask).view(-1, mask.size(-1))  # (bs * select, dim)

        target = state_S.new(state_S.size(0)).fill_(1)
        loss = F.cosine_embedding_loss(state_S, state_T, target, reduction='mean')
        return loss


class HidMseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_S, state_T, mask=None):
        '''
        Calculate the mse loss between state_S and state_T, state is the hidden state of the model

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            loss = F.mse_loss(state_S, state_T)
        else:
            mask = mask.to(state_S)
            valid_count = mask.sum() * state_S.size(-1)
            loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
        return loss


class AttCeMeanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the cross entropy  between attention_S and attention_T, the dim of num_heads is averaged

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        if len(attention_S.size()) == 4:
            attention_S = attention_S.mean(dim=1)  # (bs, len, len)
            attention_T = attention_T.mean(dim=1)
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
            loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = mask.to(attention_S)
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(1)).sum(
                dim=-1) * mask).sum() / mask.sum()
        return loss


class AttCeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the cross entropy  between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
            loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(2)).sum(
                dim=-1) * mask).sum() / mask.sum()
        return loss


class AttMseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the mse loss between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
            attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
            loss = F.mse_loss(attention_S_select, attention_T_select)
        else:
            mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()
            loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
                2)).sum() / valid_count
        return loss


class KdCeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_S, logits_T, temperature=1):
        '''
        Calculate the cross entropy between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
        return loss


class KdMseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_S, logits_T, temperature=1):
        '''
        Calculate the mse loss between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        loss = F.mse_loss(beta_logits_S, beta_logits_T)
        return loss
