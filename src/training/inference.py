import torch
from PIL import Image

# Function to generate answer
def inference_check(vision_model, processor, tokenizer, model, image, prompt="Суретте не көріп тұрсың?"):
	# Try to load image
	try:
		image = Image.open(image).convert("RGB")
	except Exception as e:
		print(f"Error at loading image {image}: {e}")

	pixel_values = processor(images=[image], return_tensors="pt").squeeze(0)
	image_features = vision_model(**pixel_values)
	image_features = model.linear_projection(image_features)

	# Tokenize input text
	text_inputs = tokenizer(
		prompt,
		max_length=128,
		truncation=True,
		padding="max_length",
		return_tensors="pt"
	)

	input_ids = text_inputs.input_ids
	attention_mask = text_inputs.attention_mask

	labels = input_ids.clone()
	labels[labels == tokenizer.pad_token_id] = -100

	# Get output 
	outs = model(pixel_values, text=input_ids, attention_mask=attention_mask, labels=labels)

	return tokenizer.decode(outs[0][outs["input_ids"].shape[-1]:])