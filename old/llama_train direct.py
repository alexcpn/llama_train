# -*- coding: utf-8 -*-
# !huggingface-cli login

import torch
from datetime import datetime
import torch._dynamo.config
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import Dataset,load_dataset
import logging as log
import os
from peft import LoraConfig, get_peft_model
from transformers import  get_linear_schedule_with_warmup # for training
from datetime import datetime
#import torch._dynamo.config
# from torch.cuda.amp import autocast,GradScaler
# from datasets import load_dataset

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

def load_texts_from_directory(directory, tokenizer, block_size=1024, keyword="<<KEYWORD>>"):
    texts = []

    for file_name in os.listdir(directory):
        if file_name.endswith(".py"):  # Filter for specific file types
            log.info(f"Adding file {file_name} for training")
            with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
                text = file.read()
                # Prepend keyword and filename info
                #text = f"This is code for project {keyword}. Filename: {file_name}\n{text}"
                
                # Split text into chunks of up to `block_size` tokens
                while len(text) > 0:
                    chunk = text[:block_size]  # Take a chunk of `block_size`
                    texts.append(chunk)
                    text = text[block_size:]  # Remove the chunk from the text

    # Tokenize each chunk
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=block_size,
        return_tensors="pt",
    )

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"], 
        "attention_mask": encodings["attention_mask"]
    })

    return dataset, encodings



def get_random_batch(len_train_data,input_ids,attention_mask,block_size=1024,
                    batch_size=12):
    """
    To get back the data
        x,y= get_random_batch(len_train_data,input_ids,attention_mask,
                        block_size=block_size,batch_size=train_batch_size)
        input_dec = tokenizer.decode(x[0,:].squeeze(), skip_special_tokens=False)
        log.info(f"Decoded x : '{input_dec}'")# correct
    """
    # random select from training data set
    ix = torch.randint(0,block_size , (batch_size,))
    x = torch.stack([(input_ids[i:i+block_size]) for i in ix])
    y = torch.stack([((attention_mask[i:i+block_size])) for i in ix])
    return x, y

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


#--------------------------------------------------------------------------------------------------------
def train_model(pmodel, input_ids, attention_mask, len_train_data, 
                 num_train_epochs, train_batch_size, block_size, device,
                 optimizer, lr_scheduler, torch_dtype):
    scaler = torch.amp.GradScaler()

    for epoch in range(num_train_epochs):
        log.info(f"Epoch {epoch+1} of {num_train_epochs}")
        epoch_loss = 0
        for i in range(0, len_train_data, block_size):
            # Get data in random per batch from input
            # not all training data may not be covered in one epoch here
            x, y = get_random_batch(len_train_data, input_ids, attention_mask,
                                    block_size=block_size, batch_size=train_batch_size)
            # attention_mask given by tokenize is array of ones= [1,1,..], that is attend to all tokens
            #with torch.amp.autocast(device_type="cuda",dtype=torch_dtype):
            # print("x.shape",x.shape)
            # print("x.shape",y.shape)
            x = x.view(-1, x.shape[-1])  # Flatten the batch and chunk dimensions
            y = y.view(-1, y.shape[-1])  # Ensure attention mask matches input shape
            outputs = pmodel(input_ids=x.to(device), attention_mask=y.to(device), labels=x.to(device))
            print("outputs.loss=",outputs.loss)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pmodel.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # Accumulate loss for logging
            epoch_loss += loss.item()
        average_epch_loss = epoch_loss/num_train_epochs
        log.info(f"Epoch {epoch} complete. Loss: {average_epch_loss}")
        
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_path = './data/nsp_troubleshooting.txt'  # a small training file to learn

# Loading the Model
############################################################################################################
# Fine-tune the model on the training data
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
    torch_dtype = torch.float16

log.info(f"Going to load the model {model_name} ")


dir_path =  "/ssd/finetune_llama/dataset"
_,encodings = load_texts_from_directory(dir_path,tokenizer)


################################################################################
# bitsandbytes parameters
################################################################################

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
############################################################################################################

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map=device_map,
)
log.info(f"Loaded Quantised model with config={bnb_config}")
############################################################################################################

################################################################################
# QLoRA parameters
################################################################################


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8, # rank
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj",
                    "up_proj","down_proj","embed_tokens","lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
)
# Now to start the training

# target_layers = [f'.{i}' for i in range(10, 20)] # first 10 to 20, next , 20 to 30 # last 30 to 32
# freeze_all__and_unfreeze_layers(model, target_layers)

# # Freeze all attention layers
# for layer in model.model.layers:  # Access each decoder layer
#     # Freeze attention submodules
#     for param in layer.self_attn.parameters():
#         param.requires_grad = False

# Apply PEFT to the model
peft_model = get_peft_model(model, peft_config)

model.train()



#===============================================================================================
# Set up the training parameters for Manual training
#===============================================================================================

encoding_shape_len = encodings["input_ids"].shape[0]
log.info(f"len_train_data={encoding_shape_len}")
block_size = 100 # for uses 24/40 gb with model loaded A100 3 bad results
assert block_size < encoding_shape_len
train_batch_size = 1 # 38 GB/40 in A100
num_train_epochs = 20
save_epochs =num_train_epochs
# Set the optimizer and learning rate scheduler
num_warmup_steps = 100
max_grad_norm = 1.0
num_train_steps = encoding_shape_len // train_batch_size * num_train_epochs
#lr_scheduler = None
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) #todo was  3e-5
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
log.info(f"len_train_data={encoding_shape_len} block_size ={block_size} batch_size= {train_batch_size}")
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
log.info(f"input_ids.shape: '{input_ids.shape}'")
x,y= get_random_batch(encoding_shape_len, input_ids,attention_mask ,
                block_size=block_size,batch_size=train_batch_size)
input_dec = tokenizer.decode(x[0, 0, :].tolist(), skip_special_tokens=False)
log.info(f"Decoded is {input_dec}")


train_model(peft_model,input_ids, attention_mask, encoding_shape_len, 
             num_train_epochs, train_batch_size, block_size, device, 
             optimizer, lr_scheduler,torch_dtype)
checkpoint_dir = f"./saved_model/llama3-final"
# log.info(f"Training over saving  model in {checkpoint_dir}")
peft_model.save_pretrained(checkpoint_dir)

log.info("Training over- Exiting normally")