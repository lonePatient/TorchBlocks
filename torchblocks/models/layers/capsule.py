import torch
import torch.nn as nn
import torch.nn.functional as F


class Capsule(nn.Module):
    def __init__(self, input_dim_capsule=1024, num_capsule=5, dim_capsule=5, routings=4):
        super(Capsule, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = self.squash
        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))

    def forward(self, x):
        u_hat_vecs = torch.matmul(x, self.W)
        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(
            0, 2, 1, 3).contiguous()  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        with torch.no_grad():
            b = torch.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = F.softmax(b, dim=1)  # (batch_size,num_capsule,input_num_capsule)
            outputs = self.activation(torch.sum(c.unsqueeze(-1) * u_hat_vecs, dim=2))  # bij,bijk->bik
            if i < self.routings - 1:
                b = (torch.sum(outputs.unsqueeze(2) * u_hat_vecs, dim=-1))  # bik,bijk->bij
        return outputs  # (batch_size, num_capsule, dim_capsule)

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale
