import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernels: list,
        num_actions: int = 7,
        bias: bool = False,
        activation_function: str = "relu",
    ):
        super(CNN, self).__init__()

        if activation_function == "gelu":
            act = nn.GELU()
        elif activation_function == "silu":
            act = nn.SiLU()
        elif activation_function == "selu":
            act = nn.SELU()
        else:
            act = nn.ReLU()

        self.channels = [in_channels] + channels
        kernels = [None] + kernels
        self.layers = nn.ModuleList()
        self.act = act

        for i in range(1, len(self.channels)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.channels[i - 1],
                        kernel_size=kernels[i],
                        out_channels=self.channels[i],
                        bias=bias,
                        padding=(kernels[i] - 1) // 2,
                    ),
                    act,
                )
            )

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(
            in_features=self.channels[-1] * 42, out_features=num_actions, bias=bias
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)

        # x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    x = torch.rand(32, 2, 6, 7)
    cnn = CNN(2, [32, 64, 128], [7, 3, 3])
    output = cnn(x)
    print(output.shape)
