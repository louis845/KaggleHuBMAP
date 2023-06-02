import torch
import numpy as np
import itertools

class PermInvLayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PermInvLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weights = torch.nn.Parameter(torch.zeros(size=(in_dim, out_dim), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(size=(out_dim,), dtype=torch.float32))

    def init_weights(self, perm_stride, perm_dims):
        assert perm_stride > 0, "perm_stride must be positive"
        assert perm_stride <= self.in_dim, "perm_stride must be less than or equal to in_dim"
        assert perm_dims > 0, "perm_dims must be positive"
        assert perm_dims <= self.in_dim, "perm_dims must be less than or equal to in_dim"
        assert perm_dims % perm_stride == 0, "perm_dims must be divisible by perm_stride"

        perm_elems = int(perm_dims / perm_stride)

        torch.nn.init.xavier_uniform_(self.weights, gain=1.0)
        torch.nn.init.zeros_(self.bias)

        with torch.no_grad():
            weights_matrix = self.weights.data

            perms = torch.tensor(np.array(list(itertools.permutations(range(perm_elems)))), dtype=torch.int64, device=self.weights.device) * perm_stride
            perms = torch.repeat_interleave(perms, repeats=perm_stride, dim=1)
            perms = perms + torch.tile(torch.arange(perm_stride, dtype=torch.int64, device=self.weights.device), dims=(perm_elems,))
            perms = torch.concat([perms,
                    torch.arange(start=perm_dims, end=self.in_dim, device=self.weights.device, dtype=torch.int64).unsqueeze(0).repeat(perms.shape[0], 1)
                                  ], dim=1)
            weights_perm_inv = torch.mean(weights_matrix[perms, :], dim=0)

            assert weights_perm_inv.shape == weights_matrix.shape, "weights_perm_inv shape does not match weights_matrix shape"

            self.weights.copy_(weights_perm_inv)

    def forward(self, x):
        """Multiply x by weights and add the bias. Acts on the last dimension as usual."""
        return torch.matmul(x, self.weights) + self.bias


if __name__ == "__main__":
    layer = PermInvLayer(10, 10)
    layer.init_weights(2, 6)

    input1 = torch.tensor([1, 0, 2, 3, -6, 4, 7, 2, 1, 3], dtype=torch.float32)
    input2 = torch.tensor([1, 0, 2, 3, -6, 4, 7, 2, 3, 1], dtype=torch.float32)
    input3 = torch.tensor([2, 3, 1, 0, -6, 4, 7, 2, 1, 3], dtype=torch.float32)
    input4 = torch.tensor([2, 3, -6, 4, 1, 0, 7, 2, 1, 3], dtype=torch.float32)

    print(layer(input1))
    print(layer(input2))
    print(layer(input3))
    print(layer(input4))