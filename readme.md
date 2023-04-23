# MeGPT: Fine-tune a Language Model with your iMessage Conversations

MeGPT allows you to fine-tune a large language model on your own messages, enabling you to talk to yourself.

<!-- add image -->

![MeGPT](/demo.png)

This repo contains code for:

1. Extracting your iMessage conversations from your Mac
1. Fine-tuning a large language model on your messages
1. Generating completions using the fine-tuned model

This is a sample repo that trains Meta AI's OPT 1.3b model with Parallel Efficient Fine-tuning (PEFT) on your iMessage conversations. You can use this repo as a starting point for fine-tuning other models on your own data.

Based off of [example code from lvwerra/trl](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/clm_finetune_peft_imdb.py)

## Usage

1.  Install the requirements:

```
pip install -r requirements.txt
```

1. On your Mac, run `extract_messages.py` to extract your iMessage conversations and save them to a CSV file:

```
python extract_messages.py
```

1. Configure `fine_tune.py` with your desired model, input CSV, and other settings. For example:

```
model_name = "facebook/opt-1.3b"
block_size = 128
input_csv = "messages.csv"
```

To see the full list of supported models, visit [PEFT Models Support Matrix.](https://github.com/huggingface/peft#models-support-matrix)

1. Train the model on your messages using fine_tune.py:

```
python fine_tune.py
```

1. Generate completions using the fine-tuned model with generate.py:

```
python generate.py
```
