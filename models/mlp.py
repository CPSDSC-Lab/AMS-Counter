from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            self.layers.append(nn.Linear(input_size, output_size))
        self.act_fn = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))
        return self.layers[-1](x)
