import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained language model
model_name = "Salesforce/codet5-large"  # You can choose "codet5-small", "codet5-base", or "codet5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the AI Assistant
class CodingAssistant:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_code(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def interact(self):
        print("AI Coding Assistant: How can I help you with your code today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = self.generate_code(user_input)
            print(f"Assistant: {response}")

# Initialize the assistant
assistant = CodingAssistant(model, tokenizer)

# Start the interaction
assistant.interact()
