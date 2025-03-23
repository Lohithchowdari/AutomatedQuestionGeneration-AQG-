model.save_pretrained("./fine_tuned_t5_model")
tokenizer.save_pretrained("./fine_tuned_t5_tokenizer")
print("Model and tokenizer saved successfully.")