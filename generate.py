import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    PeftConfig,
    PeftModel,
)

model_name = "facebook/opt-1.3b-finetuned"
config = PeftConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, model_name)

batch = tokenizer(
    "I really enjoyed the", return_tensors="pt", padding=True, truncation=True
)

model.to("cuda")


with torch.cuda.amp.autocast():
    output_tokens = model.generate(
        **batch,
        max_new_tokens=500,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
