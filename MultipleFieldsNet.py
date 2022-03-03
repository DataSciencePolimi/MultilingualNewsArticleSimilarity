from torch import Tensor
from torch import nn


class MultipleFieldsNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation_function=nn.Identity(),
                 init_weight: Tensor = None, init_bias: Tensor = None):
        super(MultipleFieldsNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, embeddings: Tensor):
        return self.activation_function(self.linear(embeddings))
