import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from evaluate import load  
import matplotlib.pyplot as plt

model_name = "./fine_tuned_t5_model"  
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5_tokenizer")  
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("microsoft/ms_marco", "v1.1")

def preprocess_function(examples):
    inputs = []
    for passage in examples['passages']:
        passage_text = passage[0]['passage_text'] if isinstance(passage, list) and len(passage) > 0 else ""
        inputs.append("generate question: " + passage_text)
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer([answer[0] if answer else "" for answer in examples['answers']], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./question_generation",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,  
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./fine_tuned_t5_model")
tokenizer.save_pretrained("./fine_tuned_t5_tokenizer")
print("Model and tokenizer saved successfully.")

test_dataset = dataset["test"].map(preprocess_function, batched=True)
eval_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Evaluation results: {eval_results}")

train_loss = []
eval_loss = []
for log in trainer.state.log_history:
    if "loss" in log and "eval_loss" not in log:  
        train_loss.append(log["loss"])
    elif "eval_loss" in log:  
        eval_loss.append(log["eval_loss"])

epochs = range(1, min(len(train_loss), len(eval_loss)) + 1)

plt.plot(epochs, train_loss[:len(epochs)], label="Training Loss")
plt.plot(epochs, eval_loss[:len(epochs)], label="Evaluation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss per Epoch")
plt.legend()
plt.show()