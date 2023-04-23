from itertools import chain

import os
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


model_name = "facebook/opt-1.3b"
block_size = 128
input_json = "response.json"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name, model_max_length=1024, padding_side="left"
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


model = prepare_model_for_int8_training(model)
for name, param in model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False
    if getattr(model, "is_loaded_in_8bit", False):
        if param.ndim == 1 and "layer_norm" in name:
            param.data = param.data.to(torch.float16)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


target_modules = None
if "gpt-neox" in model_name:
    target_modules = [
        "query_key_value",
        "xxx",
    ]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


training_args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    logging_strategy="steps",
)

data = load_dataset("json", data_files=input_json, split="train")
data = data.map(lambda x: {"label": 0})
data = data.train_test_split(test_size=0.1)


def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


columns = data["train"].features
data = data.map(
    lambda samples: tokenizer(samples["text"], padding=True, truncation=True),
    batched=True,
    remove_columns=columns,
    num_proc=os.cpu_count(),
)

data = data.map(group_texts, batched=True, batch_size=1000, num_proc=os.cpu_count())
data.set_format(type="torch", columns=["input_ids", "labels"])

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# save the model
model.save_pretrained(
    f"{model_name}-finetuned",
    save_function=trainer.save_model,
    push_to_hub=False,
)

model.config.use_cache = True  # re-enable for inference

# inference
PROMPT = """hello, how are you?"""
batch = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)

with torch.cuda.amp.autocast():
    output_tokens = model.generate(
        **batch,
        max_new_tokens=500,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
