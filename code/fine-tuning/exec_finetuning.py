from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Generate a training file
training_file = client.files.create(
     file=open("./data/in-domain/bin0/fine_tuning_data/train-set-finetuning-promptft1.json", "rb"),
     purpose="fine-tune"
   )
training_file_id = training_file.id

# Fine-tuning job
client.fine_tuning.jobs.create(
  training_file=training_file_id,
  model="gpt-4.1-nano-2025-04-14",
  # method= {
  #   "type": "supervised",
  #   "supervised": {
  #     "hyperparameters": {
  #       "batch_size": "2",
  #       "learning_rate_multiplier": "1.8",
  #       "n_epochs": "3",
  #     }
  #   }
  # },
)


