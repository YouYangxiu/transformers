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

prompt = "Please summary the following essay(s):"
# q1 = """New Year's Day officially begins as soon as people yell "Happy New Year" at midnight. Most people continue partying well after midnight into the wee hours of the first day of the new year. In fact, many New Year's parties include breakfast or brunch. Sometimes at the stroke of midnight there will be fireworks and couples often kiss. One of the most famous New Year's celebrations takes place in New York City's Time Square, where a huge cut crystal ball drops at midnight in front of millions of people standing in the cold. Many more millions watch on television. Some groups, called Polar Bear Clubs, jump into the cold ocean water on New Year's Day as a literal way to start the new year fresh. \n"""
# q2 = """The Braille system also had important cultural effects beyond the sphere of written culture. Its invention later led to the development of a music notation system for the blind, although Louis Braille did not develop this system himself (Jimenez, et al., 2009). This development helped remove a cultural obstacle that had been introduced by the popularization of written musical notation in the early 1500s. While music had previously been an arena in which the blind could participate on equal footing, the transition from memory-based performance to notation-based performance meant that blind musicians were no longer able to compete with sighted musicians (Kersten, 1997). As a result, a tactile musical notation system became necessary for professional equality between blind and sighted musicians (Kersten, 1997). \n"""
# q3 = """Lennon wrote, or co-wrote some of the most memorable tunes ever written in the Rock 'n' Roll genre. He found great success as a member of the Beatles, which is the most commercially successful group in history, and as a solo artist. Lennon was also a political activist. His views on religion and politics caused a great deal of controversy in the United States. He was an outspoken critic of the country's involvement in the Vietnam conflict. His popularity caused great concern with government officials because he had the attention of the young people in America at the time."""

q1 = "I am"

q2 = "a helpful"

q3 = "assistant."

# messages = [{"role": "user", "content": prompt + q1 + q2}]
#
# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

text_len = len(tokenizer(prompt)["input_ids"])
query1_len = len(tokenizer(q1)["input_ids"])
query2_len = len(tokenizer(q2)["input_ids"])
query3_len = len(tokenizer(q3)["input_ids"])

os.environ["text_len"] = str(text_len)
os.environ["query1_len"] = str(query1_len)
os.environ["query2_len"] = str(query2_len)
os.environ["query3_len"] = str(query3_len)

model_inputs = tokenizer([prompt + q1 + q2 + q3], return_tensors="pt").to("cuda:0")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=False, top_p=None)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)


