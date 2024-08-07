# -*- coding: utf-8 -*-
"""
@Time: 2024/08/06/16/09
@Author: josephyou
@Email: josephyou@tencent.com
"""
from sympy.physics.optics import lens_formula

import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置只使用 GPU 0
transformers.logging.set_verbosity_error()

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

prompt = "Please comment the following essay."
q1 = """The invention of Braille was a major turning point in the history of disability. The writing system of raised dots used by visually impaired people was developed by Louis Braille in nineteenth-century France. In a society that did not value disabled people in general, blindness was particularly stigmatized, and lack of access to reading and writing was a significant barrier to social participation. The idea of tactile reading was not entirely new, but existing methods based on sighted systems were difficult to learn and use. As the first writing system designed for blind people’s needs, Braille was a groundbreaking new accessibility tool. It not only provided practical benefits, but also helped change the cultural status of blindness. This essay begins by discussing the situation of blind people in nineteenth-century Europe. It then describes the invention of Braille and the gradual process of its acceptance within blind education. Subsequently, it explores the wide-ranging effects of this invention on blind people’s social and cultural lives."""
q2 = """The Braille system also had important cultural effects beyond the sphere of written culture. Its invention later led to the development of a music notation system for the blind, although Louis Braille did not develop this system himself (Jimenez, et al., 2009). This development helped remove a cultural obstacle that had been introduced by the popularization of written musical notation in the early 1500s. While music had previously been an arena in which the blind could participate on equal footing, the transition from memory-based performance to notation-based performance meant that blind musicians were no longer able to compete with sighted musicians (Kersten, 1997). As a result, a tactile musical notation system became necessary for professional equality between blind and sighted musicians (Kersten, 1997)."""

messages = [{"role": "user", "content": prompt + q1 + q2}]
#
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

text_len = len(tokenizer(text)["input_ids"])
q1_len = len(tokenizer(q1)["input_ids"])
q2_len = len(tokenizer(q2)["input_ids"])

model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=100, do_sample=False, top_p=None)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
