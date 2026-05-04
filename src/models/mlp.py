import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to create Multi Layer Perceptron
def load_mlp(input_dim: int = 4608, hidden_dim: int =  16872, output_dim: int = 4080, device: str = "cpu"):
	class MLP(nn.Module):
		def __init__(self, input_dim, hidden_dim, output_dim):
			super().__init__()
			self.input_dim = input_dim
			self.hidden_dim = hidden_dim
			self.output_dim = output_dim

			self.first_linear = nn.Linear(input_dim, hidden_dim)
			self.second_linear = nn.Linear(hidden_dim, output_dim)
		
		def forward(self, x):
			x = F.relu(self.first_linear(x))
			x = F.relu(self.second_linear(x))

			return x
	
	# Initialize the model
	mlp = MLP(input_dim, hidden_dim, output_dim).to(device)

	return mlp