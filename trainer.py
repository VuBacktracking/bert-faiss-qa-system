from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

from utils.preprocess import preprocess_training_examples, preprocess_validation_examples
from utils.metric import compute_metrics
import yaml

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["Config"]["MODEL_NAME"]
MAX_LENGTH = config["Config"]["MAX_LENGTH"]
STRIDE = config["Config"]["STRIDE"]
DATASET_NAME = config["Config"]["DATASET_NAME"]

raw_datasets = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# SET UP DATASET
train_dataset = raw_datasets["train"].map(
          preprocess_training_examples ,
          batched=True, 
          remove_columns=raw_datasets["train"].column_names ,
          )

validation_dataset = raw_datasets["validation"].map(
      preprocess_validation_examples,
      batched=True, 
      remove_columns=raw_datasets["validation"].column_names ,
)

args = TrainingArguments(
    output_dir="distilbert-finetuned-squadv2", # Directory to save output
    evaluation_strategy="no", # Do not evaluate automatically after each epoch
    save_strategy="epoch", # Save checkpoint after each epoch
    learning_rate=2e-5, # Learning rate
    num_train_epochs=3, # Number of training epochs
    weight_decay=0.01, # Weight decay to prevent overfitting
    fp16=True, # Use half-precision data type to optimize resources
    push_to_hub=True, # Push training results to HuggingFace Hub
)

# Initialize a Trainer object for training the model
trainer = Trainer(
      model=model, # Use the pre-trained model
      args=args, # Training parameters and configurations
      train_dataset=train_dataset, # Use the training dataset
      eval_dataset=validation_dataset, # Use the evaluation dataset
      tokenizer=tokenizer, # Use the tokenizer to process text 
      )
# Start the training process
trainer.train()

# EVALUATE THE MODEL

# Load the "squad" metric from the evaluate library
metric = evaluate.load("squad_v2")

# Perform predictions on the validation dataset
predictions , _, _ = trainer.predict(validation_dataset)

# Get the start and end logits of the predicted answers
start_logits , end_logits = predictions
# Calculate evaluation metrics using the compute_metrics function
results = compute_metrics(
              start_logits ,
              end_logits ,
              validation_dataset ,
              raw_datasets["validation"]
            )

# Print the evaluation results
print(results)

trainer.push_to_hub(commit_message="Training complete")