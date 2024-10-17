import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

# Load a pre-trained Command R model and tokenizer
model_name = "command-r-model-name"  # Replace with actual Command R model identifier
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Your few-shot examples formatted as input-output pairs
few_shot_examples_crosslingual = {
    'ar': [{'role': 'User', 'message': 'Write your answer in Turkish. How should I choose what cheese to buy?'},
           {'role': 'Chatbot', 'message': 'Pek çok farklı peynir türü vardır, bu nedenle hangi peynirin satın alınacağına karar vermek kişisel tercihe, bulunabilirliğe ve kullanım amacına bağlıdır.'}],
    'de': [{'role': 'User', 'message': 'Write your answer in Korean. How should I choose what cheese to buy?'},
           {'role': 'Chatbot', 'message': '치즈의 종류는 다양하므로 구매할 치즈를 선택하는 것은 개인 취향, 가용성 및 용도에 따라 다릅니다.'}],
    # Add more examples as needed...
}

# Prepare data for training (flattened into input-output pairs)
def prepare_data(few_shot_data):
    inputs, outputs = [], []
    for lang, examples in few_shot_data.items():
        for conversation in examples:
            if conversation['role'] == 'User':
                user_message = conversation['message']
            elif conversation['role'] == 'Chatbot':
                chatbot_reply = conversation['message']
                inputs.append(user_message)
                outputs.append(chatbot_reply)
    return inputs, outputs

# Prepare the dataset
inputs, outputs = prepare_data(few_shot_examples_crosslingual)
train_data = Dataset.from_dict({
    'input_text': inputs,
    'target_text': outputs
})

# Tokenize dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_data = train_data.map(preprocess_function, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
)

# Create Trainer object
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("tuned_command_r_model")

# Example Inference
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with a new input
input_text = "Write your answer in Spanish. How should I choose what cheese to buy?"
print(generate_response(input_text))
