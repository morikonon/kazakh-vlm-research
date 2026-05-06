import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs, config):
	model.train()
	criterion = nn.CrossEntropyLoss()
	device = config.device
	for epoch in range(config.num_epochs):
		for index, batch in enumerate(train_loader):
			img, mask, labels = batch
			img, mask, labels = img.to(device), mask.to(device), labels.to(device)

			outs = model(img, mask, labels)

			loss = criterion(outs, labels)

			losses += loss.item()
	
	print(f"Epoch: {epoch}| Loss: {losses / len(train_loader)}")

	
