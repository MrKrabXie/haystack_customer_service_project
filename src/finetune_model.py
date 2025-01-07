from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer


def finetune():
    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 加载数据集
    dataset = load_dataset('glue','sst2')

    # 数据预处理
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=True,  # 这将在训练结束时自动将模型推送到 Hugging Face Hub
        hub_model_id="CrabWade/my_first_model_repo",  # 替换为你的用户名和仓库名
        hub_token="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"  # 替换为你的 Hugging Face Token
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # 开始训练
    trainer.train()


def predict():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # 加载微调后的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("CrabWade/my_first_model_repo")
    model = AutoModelForSequenceClassification.from_pretrained("CrabWade/my_first_model_repo")

    # 示例文本
    text = "This movie was really great!"
    inputs = tokenizer(text, return_tensors="pt")

    # 预测
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1)
    print(f"预测类别: {predicted_class.item()}")


if __name__ == "__main__":
    finetune()
    predict()