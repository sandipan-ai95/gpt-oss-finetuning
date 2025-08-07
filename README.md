# GPT-OSS-20B Medical Reasoning Chatbot

This repository contains a fine-tuned version of `openai/gpt-oss-20b`, adapted to provide medically grounded answers in response to user queries. The model is designed to simulate **clinical reasoning** based on natural language questions, especially useful for educational, triage, and general health information purposes.

---

## What’s Inside

- Fine-tuned LoRA adapters for **medical chat reasoning**
- OpenAI-style multi-turn **chat inference** using `apply_chat_template`
- Inference script for generating responses to health-related queries
- Notebook](gpt-oss-medical.ipynb) that:
  - Loads base + PEFT model
  - Merges and runs generation
  - Supports multi-language prompts and configurable reasoning language

---

## Example Usage

```python
# Example prompt
USER_PROMPT = "What medicines should I take for liver diseases?"

# Messages list 
messages = [
    {"role": "user", "content": USER_PROMPT}
]

# Generate response
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output_ids = model.generate(inputs, max_new_tokens=512)
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
