from airllm import AutoModel
import mlx.core as mx

model = AutoModel.from_pretrained("mlx-community/Meta-Llama-3.1-405B-Instruct-8bit",
    #compression='4bit'
    )

input_text = [
        'Tell me about the history of the universe.'
    ]

MAX_LENGTH = 128
input_tokens = model.tokenizer(input_text,
    return_tensors="np", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)

input_tokens

generation_output = model.generate(
    mx.array(input_tokens['input_ids']), 
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

print(generation_output)