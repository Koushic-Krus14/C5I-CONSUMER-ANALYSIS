from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load pretrained model and tokenizer
model_name = 'EleutherAI/gpt-neo-125M'  # or 'EleutherAI/gpt-j-6B' for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Summarize the main points from the following text"
response = query_llm(prompt)
print(response)
