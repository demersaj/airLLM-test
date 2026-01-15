from airllm import AutoModel
import mlx.core as mx
import time

model = AutoModel.from_pretrained("mlx-community/Meta-Llama-3.1-405B-Instruct-8bit",
    compression='None'
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

# measure time to first token by generating 1 token first
start_time = time.perf_counter()
first_token_output = model.generate(
    mx.array(input_tokens['input_ids']), 
    max_new_tokens=1,
    use_cache=True,
    return_dict_in_generate=True)
first_token_time = time.perf_counter()

# now generate the full sequence
full_start_time = time.perf_counter()
generation_output = model.generate(
    mx.array(input_tokens['input_ids']), 
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)
full_end_time = time.perf_counter()

# calculate metrics
time_to_first_token = first_token_time - start_time
total_time = full_end_time - full_start_time

# count generated tokens (excluding input tokens)
input_token_count = len(input_tokens['input_ids'][0])

# get generated tokens from output
if hasattr(generation_output, 'sequences'):
    total_token_count = len(generation_output.sequences[0])
elif isinstance(generation_output, dict) and 'sequences' in generation_output:
    total_token_count = len(generation_output['sequences'][0])
else:
    # fallback: estimate based on max_new_tokens
    total_token_count = input_token_count + 3

generated_token_count = total_token_count - input_token_count
tokens_per_second = generated_token_count / total_time if total_time > 0 else 0

# output generation output and performance metrics
print(generation_output)
print("\n================================================")
print(f"\n--- Performance Metrics ---")
print(f"Time to first token: {time_to_first_token*1000:.2f} ms")
print(f"Total generation time: {total_time:.3f} s")
print(f"Generated tokens: {generated_token_count}")
print(f"Tokens per second: {tokens_per_second:.2f}")