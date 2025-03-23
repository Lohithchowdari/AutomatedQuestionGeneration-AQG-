from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = load_dataset("ms_marco", 'v1.1', split="train")

context = dataset[0]['passage']

input_ids = tokenizer.encode(context, return_tensors='pt', max_length=512, truncation=True)

outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Question: {question}")