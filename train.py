from setfit import SetFitModel, sample_dataset, TrainingArguments, Trainer
import json 
from datasets import DatasetDict, Dataset

model_name = "BAAI/bge-small-en-v1.5"
model = SetFitModel.from_pretrained(model_name)

data = json.load(open("dataset.json", mode='r',encoding="utf-8"))
train_dataset = Dataset.from_dict(data['train'])
test_dataset = Dataset.from_dict(data['test'])
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=16)

model.labels = ["cita", "comprar", "trabajo", "recordatorio", "estudios", "hogar"]

args = TrainingArguments(
    batch_size=16,
    num_epochs=50,
    max_steps=20
)

trainer = Trainer(
    model=model,
    args=args, 
    train_dataset=train_dataset
    )
trainer.train()
print(trainer.evaluate(test_dataset))
model.save_pretrained("dailynoteclassifier-setfit-v1.5-16-shot")
