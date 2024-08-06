# -*- coding: utf-8 -*-
"""
@Time: 2024/08/06/16/09
@Author: josephyou
@Email: josephyou@tencent.com
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 设置只使用 GPU 0
transformers.logging.set_verbosity_error()


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

prompt = "Give me a short introduction to large language model."

messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=False, top_p=None)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)