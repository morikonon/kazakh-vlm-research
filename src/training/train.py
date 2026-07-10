import torch
from tqdm import tqdm


def run_epoch(model, loader, device, optimizer=None):
	model.train(optimizer is not None)

	total_loss = 0.0
	with torch.set_grad_enabled(optimizer is not None):
		for batch in tqdm(loader, leave=False):
			pixel_values = batch["pixel_values"].to(device)
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			labels = batch["labels"].to(device)

			outs = model(pixel_values, text=input_ids, attention_mask=attention_mask, labels=labels)
			loss = outs.loss

			if optimizer is not None:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			total_loss += loss.item()

	return total_loss / len(loader)


def train_model(model, train_loader, val_loader, optimizer, config, checkpoint_path="best_model.pt"):
	device = config.device
	model.to(device)

	best_val_loss = float("inf")

	for epoch in range(config.num_epochs):
		train_loss = run_epoch(model, train_loader, device, optimizer)
		val_loss = run_epoch(model, val_loader, device)

		print(f"Epoch {epoch + 1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model.state_dict(), checkpoint_path)

	return model
