import torch.nn as nn


class ConvLoRA(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.conv = conv
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation

        self.encoder = nn.Conv2d(
            in_channels, rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False
        )
        self.decoder = nn.Conv2d(
            rank, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False
        )

        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.decoder.weight)

        self.active = False

    def activate(self):
        self.active = True
        self.encoder.weight.requires_grad = True
        self.decoder.weight.requires_grad = True

    def deactivate(self):
        self.active = False
        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False
        nn.init.zeros_(self.decoder.weight)

    def forward(self, x):
        out = self.conv(x)
        if self.active:
            out = out + self.decoder(self.encoder(x))
        return out


def build_conv_lora_queue(model: nn.Module, rank: int = 4):
    queue = []
    replace_count = 0

    def replace_conv(module: nn.Module):
        nonlocal replace_count
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                lora = ConvLoRA(child, rank=rank)
                setattr(module, name, lora)
                queue.append(lora)
                replace_count += 1
            else:
                replace_conv(child)

    replace_conv(model)
    queue.reverse()
    print(f"替换完成：共 {replace_count} 个 3x3 卷积层（已反转，高层优先）")
    return queue


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if 'encoder' in name or 'decoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
