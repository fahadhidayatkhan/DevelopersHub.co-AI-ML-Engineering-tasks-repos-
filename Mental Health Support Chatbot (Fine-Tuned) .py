# Task 5: Mental Health Support Chatbot (Fine-Tuned)

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Load dataset (EmpatheticDialogues)
dataset = load_dataset("empathetic_dialogues")

# 2. Choose base model (DistilGPT2 for lightweight fine-tuning)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Tokenize data
def tokenize(batch):
    return tokenizer(batch["utterance"], truncation=True, padding="max_length", max_length=64)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4. Training setup
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

# 5. Fine-tune
trainer.train()

# 6. Save model
trainer.save_model("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")

# 7. Simple CLI interface
def chat():
    print("💬 Mental Health Support Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", response)

chat()