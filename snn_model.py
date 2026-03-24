import torch
import torch.nn as nn
import snntorch as snn

class SNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(64, num_classes)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for step in range(x.size(1)):  # time steps
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=1)