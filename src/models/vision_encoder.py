from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

def load_vision_encoder(model_name: str = "google/siglip-so400m-patch14-384", device: str = "cpu"):
	# Load model and its processor
	vision_encoder = AutoModel.from_pretrained(model_name).to(device)
	processor = AutoProcessor.from_pretrained(model_name)

	return vision_encoder, processor