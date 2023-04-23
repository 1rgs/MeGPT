import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import (
    PeftConfig,
    PeftModel,
)
import re

model_name = "facebook/opt-1.3b-finetuned"
config = PeftConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, model_name)
model.to("cuda")

stop_words_ids = [tokenizer.encode(stop_word) for stop_word in ["MeGPT:", "person:"]]


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self),

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]
    ):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]


stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


def chat_with_model(input_text, conversation_history):
    conversation_text = f"{conversation_history}person: {input_text}\nMeGPT: "

    batch = tokenizer(conversation_text, return_tensors="pt", padding=True)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            **batch,
            max_new_tokens=100,
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            do_sample=True,
            stopping_criteria=stopping_criteria,
        )

    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if "  " in response_text:
        regex = re.compile(r"\s{2,}")
        split = regex.split(response_text)
        response_text = split[1]

    conversation_history += f"person: {input_text}\nMeGPT: {response_text}\n"

    return response_text, conversation_history


conversation_history = "MeGPT: hi i am you. How can I help?\n"
print(conversation_history)
while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        break
    response, conversation_history = chat_with_model(user_input, conversation_history)

    print("MeGPT:", response)
