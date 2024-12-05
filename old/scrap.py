
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


def freeze_all__and_unfreeze_layers(model, target_layers):
    """Freezes all layers of the model and then unfreezes specific target layers.

    Args:
        model: The model to freeze and unfreeze layers.
        target_layers: A list of layer names to unfreeze.
    """

    # Freeze all layers
    for name, param in model.named_parameters():
        param.requires_grad = False

     # Unfreeze target layers
    for name, param in model.named_parameters():
        # log.info(f"Parameter {name}")
        # if any(layer in name for layer in target_layers)
        if any(layer in name for layer in target_layers) and 'input_layernorm' in name:
            log.info(f"Unfreezing {name}")
            param.requires_grad = True

    # Verify which layers are trainable
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.info(
                f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")


