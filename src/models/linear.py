import torch
import torch.nn as nn

def load_linear(input_dim: int = 1152, output_dim: int = 4080, device: str = "cpu"):
	class Linear(nn.Module):
		def __init__(self, input_dim, output_dim):
			super().__init__()

			self.linear = nn.Linear(input_dim, output_dim)
		def forward(self, x):
			x = self.linear(x)
			return x
	
	linear = Linear(input_dim, output_dim).to(device)

	return linear
