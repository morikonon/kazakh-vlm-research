import os
import json
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def get_dataset(tokenizer, processor, json_path: str = None, images_root: str = "/coco_karpathy", max_length: int = 128)
	class VLMDataset(Dataset):
		def __init__(self, json_path, images_root, tokenizer, processor, max_length=128):
			self.images_root = images_root
			self.tokenizer = tokenizer
			self.processor = processor
			self.max_length = max_length

			print(f"Loading data from {json_path}...")
			with open(json_path, 'r', encoding='utf-8') as f:
				self.data = json.load(f)

			# Filter valid entries
			self.data = [entry for entry in self.data if entry.get('captions_kk')]
			print(f"Dataset size: {len(self.data)}")

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			entry = self.data[idx]
			image_path = os.path.join(self.images_root, entry['image_path'])

			# 1. Image Loading with fallback
			try:
				image = Image.open(image_path).convert('RGB')
			except:
				return self.__getitem__(random.randint(0, len(self.data)-1))

			# 2. Process Image
			# SigLIP processor usually returns [1, 3, H, W], we need [3, H, W]
			pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

			# 3. Text Processing
			caption = random.choice(entry['captions_kk'])
			prompt_text = "Суреттегі көрініс: "

			# Tokenize separately to know exact lengths
			# Add EOS to caption manually if tokenizer doesn't add it automatically in this flow
			full_text = prompt_text + caption + self.tokenizer.eos_token

			tokenized = self.tokenizer(
				full_text,
				max_length=self.max_length,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			)

			input_ids = tokenized.input_ids.squeeze(0)
			attention_mask = tokenized.attention_mask.squeeze(0)
			labels = input_ids.clone()

			# 4. Smart Masking
			# Mask everything that is NOT the caption response
			# We calculate length of prompt tokens roughly
			prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
			prompt_len = len(prompt_tokens)

			# Assuming format [BOS, Prompt_Tokens, Caption_Tokens, EOS, PAD...]
			# We mask [0 : prompt_len + 1] (including BOS)
			# Note: This is heuristic. Check generated labels visually if possible.
			labels[:prompt_len + 1] = -100
			labels[labels == self.tokenizer.pad_token_id] = -100

			return {
				"pixel_values": pixel_values,
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"labels": labels
			}
	
	dataset = VLMDataset(
		json_path=json_path,
		images_root=images_root,
		tokenizer=tokenizer,
		processor=processor,
		max_length=max_length
	)

	return dataset

def split_data(dataset, train_size: int = 0.9, val_size: int = 0.05 test_size: int = 0.05):
	train_size = int(train_size * len(dataset))
	val_size = int(val_size * len(dataset))
	test_size = len(dataset) - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(
		dataset, [train_size, val_size, test_size]
	)

	return train_dataset, val_dataset, test_size

def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size: int = 1):
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader