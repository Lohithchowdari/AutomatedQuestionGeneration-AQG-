from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

dataset = load_dataset("microsoft/ms_marco", "v1.1")

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = ["generate question: " + passage['passage_text'][0] for passage in examples['passages']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")  # Ensures consistent input length

    with tokenizer.as_target_tokenizer():
        labels = tokenizer([answer[0] if answer else "" for answer in examples['answers']], max_length=128, truncation=True, padding="max_length")  # Ensures consistent label length

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./question_generation",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator  
)

trainer.train()