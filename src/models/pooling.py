import math
import torch
import torch.nn as nn
from mlp import load_mlp

# Function to create Pooling Layer
def load_pooling_layer(input_dim: int = 1152, output_dim: int = 4080, device: str = "cpu"):
	class Pooling(nn.Module):
		def __init__(self, input_dim, output_dim):
			super().__init__()
			self.input_dim = input_dim
			self.output_dim = output_dim

			self.mlp = load_mlp(input_dim=input_dim, hidden_dim=int(4 * input_dim), output_dim=output_dim, device=device)
		
		def forward(self, x):
			# Batch, Length, Dimension - Batch size, 729, 1152
			B, L, D = x.shape()

			# math.sqrt(729) = 27
			height, width = math.sqrt(L), math.sqrt(L)

			# (27 - 1) / 2 = 13
			new_height, new_width = (height - 1) / 2, (width - 1) / 2

			# We make like this: (Batch, 729, 1152) -> (Batch, 13, 13, 2, 2, 1152)
			x = x.view(B, new_height, new_width, 2, 2, D)

			# Then we make this operation: (Batch, 13, 13, 2, 2, 1152) -> (Batch, 169, 4608)
			x = x.reshape(B, new_height * new_width, 2 * 2 * D)

			# Use MLP layer
			x = self.mlp(x)

			return x
	
	# Initialize pooling model
	pooling = Pooling(input_dim, output_dim).to(device)

	return pooling

	









