from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import load_peft_model, LoraConfig

# Function to Load Large Language Models
# We choose KazLM with 8 billions parameters from ISSAI
def load_llm(configs, model_name: str = "issai/LLama-3.1-KazLLM-1.0-8B", device: str = "cpu", LoRA: bool = False):

	# Initialize Bits And Bytes Configuration
	bitsandconfig = BitsAndBytesConfig(
		load_in_4bit=True
	)
	# Load model and tokenizer
	# Also we set the model to device
	llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bitsandconfig).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	tokenizer.pad_token_id = tokenizer.eos_token_id

	# If do not want Low Rank Adaption we return withour LoRA
	if LoRA == False:
		return llm, tokenizer

	# Else we initialize LoRA configuration 
	else:

		# Lora Configuration
		LoraConfig = LoraConfig(
			lora_rank=configs.lora_rank,
			target_modules=["q_proj", "k_proj", "v_proj"]
		)

		# Get lora model
		lora_model = load_peft_model(llm, LoraConfig)

		# Return LoRA model and tokenizer
		return lora_model, tokenizer