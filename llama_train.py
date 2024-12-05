# -*- coding: utf-8 -*-
# !huggingface-cli login

import torch
from datetime import datetime
import torch._dynamo.config
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import Dataset
import logging as log
import os
from peft import LoraConfig, get_peft_model
from datetime import datetime


log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
time_hash = str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
print(torch.__version__)
outfile = log_directory + "/llama32_" + time_hash + '.log'


log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])


def create_prompt(question):
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant to help users<|eot_id|><|start_header_id|>user<|end_header_id|>

    What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    system_message = "You are a helpful assistant.Please answer the question if it is possible"

    prompt_template = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
    # print(prompt_template)
    return prompt_template
# --------------------------------------------------------------------------------------------------------
# Define a function to process prompts
# --------------------------------------------------------------------------------------------------------
def process_prompt(prompt, model, tokenizer, device, max_length=250):
    """Processes a prompt, generates a response, and logs the result."""
    prompt_encoded = tokenizer(
        prompt, truncation=True, padding=False, return_tensors="pt")
    model.eval()
    output = model.generate(
        input_ids=prompt_encoded.input_ids.to(device),
        max_length=max_length,
        attention_mask=prompt_encoded.attention_mask.to(device)
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    log.info("-"*80)
    log.info(f"Model Question:  {prompt}")
    log.info("-"*80)
    log.info(f"Model answer:  {answer}")
    log.info("-"*80)

        
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<|finetune_right_pad_id|>" # https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
log.info(f"pad_token_id={tokenizer.pad_token_id} / {tokenizer.decode(tokenizer.pad_token_id)}") #prints 0

# --------------------------------------------------------------------------------------------------------
#  Split into smaller tokens for model to train in smaller GPUs
# --------------------------------------------------------------------------------------------------------
def split_sequences(examples, block_size=512, stride=128, pad_token_id=0):
    """
    Splits the tokenized sequences into smaller chunks and pads shorter chunks.

    Args:
        examples: Dictionary of tokenized examples (containing 'input_ids' and 'attention_mask').
        block_size: The desired length of each chunk.
        stride: The number of tokens to overlap between consecutive chunks.
        pad_token_id: The token ID used for padding.

    Returns:
        A dictionary with the split sequences.
    """
    # Concatenate all examples for each key
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Split by chunk of block_size with overlap of stride
    result = {
        k: [
            t[i : i + block_size]
            if len(t[i : i + block_size]) == block_size
            else t[i : i + block_size] + [pad_token_id] * (block_size - len(t[i : i + block_size]))
            for i in range(0, total_length, stride)
        ]
        for k, t in concatenated_examples.items()
    }

    # Copy input_ids to labels for language modeling tasks
    result["labels"] = result["input_ids"].copy()
    return result

# --------------------------------------------------------------------------------------------------------
# Loading the Model
# --------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name}")
bf16 = False
fp16 = True

major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0}  # lets load on the next
# device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype = torch.bfloat16
else:
    # it is not possible to train in float16  #https://github.com/huggingface/transformers/issues/23165
    torch_dtype = torch.float32

log.info(f"Going to load the model {model_name} ")

dir_path =  "./dataset"

# --------------------------------------------------------------------------------------------------------
# Load custom files to a dict
# --------------------------------------------------------------------------------------------------------

def load_code_files(dir_path):
    data = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".py"):
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "r") as f:
                file_content = f.read()
                data.append({"text": file_content})
    return data

data = load_code_files(dir_path)
dataset = Dataset.from_list(data)

# Now you can proceed with tokenization
log.info(f"Number of files loaded:= {len(dataset)}")
# print("Raw text example:")
# print(dataset[0]["text"])  # Print the first file's content

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Print the number of examples being processed
    print(f"Processing {len(examples['text'])} examples")
    
    # Use a longer max_length or remove truncation to see full content
    tokenized = tokenizer(
        examples["text"],
        truncation=False,  # Remove truncation temporarily
        padding=True,
        max_length=None,   # Remove max_length constraint
        return_tensors="pt",
    )
    
    return tokenized

# Tokenize with detailed logging
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)
log.info(f"tokenized_dataset={tokenized_dataset}")
log.info(len(tokenized_dataset['input_ids'][0]))

# Make the tokens list smaller

block_size = 512  # Adjust as needed
stride = 128     # Adjust as needed

processed_dataset = tokenized_dataset.map(
    lambda examples: split_sequences(examples, block_size, stride,tokenizer.pad_token_id),
    batched=True,
)

log.info(f"processed_dataset={processed_dataset} Total Rows= {len(processed_dataset['input_ids'])}")
log.info(f"processed_dataset max lenfor row 0 = {len(processed_dataset['input_ids'][0])}")
# for i in range(0,len(processed_dataset['input_ids'])):
#     assert len(processed_dataset['input_ids'][i]) == block_size

# --------------------------------------------------------------------------------------------------------
# bitsandbytes parameters
# --------------------------------------------------------------------------------------------------------

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# --------------------------------------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map=device_map,
)
log.info(f"Loaded Quantised model with config={bnb_config}")


# --------------------------------------------------------------------------------------------------------
# Load LoRA configuration
# Parameters from https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
#  learning_rate=3e-5,  # Learning rate for training
# --------------------------------------------------------------------------------------------------------

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8, # rank
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj",
                    "up_proj","down_proj","embed_tokens","lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
)

# --------------------------------------------------------------------------------------------------------
# Apply PEFT to the model
# --------------------------------------------------------------------------------------------------------
peft_model = get_peft_model(model, peft_config)

# --------------------------------------------------------------------------------------------------------
# Train via Hugging Face Libraries
# --------------------------------------------------------------------------------------------------------
model.train() # set model to train
training_args = TrainingArguments(
    output_dir="./models/",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=1,  # number of training epochs
    per_device_train_batch_size=1,  # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    learning_rate=3e-5,  # Learning rate for training
    weight_decay=0.1,  # Weight decay to regularize training
    gradient_accumulation_steps=4,
    warmup_steps=500,
    max_grad_norm= 1.0,
    prediction_loss_only=True,
    logging_strategy="epoch",
    logging_steps=10,
    #report_to="tensorboard",
    save_total_limit=1,  # do not save the model
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset,
    # eval_dataset=test_dataset,
)

# --------------------------------------------------------------------------------------------------------
# Start the training
# --------------------------------------------------------------------------------------------------------

trainer.train()

checkpoint_dir = f"./saved_model/llama3-final"
# log.info(f"Training over saving  model in {checkpoint_dir}")
peft_model.save_pretrained(checkpoint_dir)

log.info("Training over- Exiting normally")
