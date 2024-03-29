from datasets import load_dataset
import evaluate
import numpy as np
from transformers import MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import numpy as np


MODEL = "vinai/vinai-translate-en2vi"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
# mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="vi_VN")
tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="en_XX", tgt_lang="vi_VN")
ds = load_dataset("quan246/MultiMed_final")
metric = evaluate.load("sacrebleu")


# preprocess functions
source_lang = "en"
target_lang = "vi"

def preprocess_function(examples):
    inputs = [example[source_lang].replace(" \n", "").strip() for example in examples["translation"]]
    targets = [example[target_lang].replace(" \n", "").strip() for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
tokenized_en_vi = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


training_args = Seq2SeqTrainingArguments(
    output_dir="vinai_en_vi",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    push_to_hub=True,
    predict_with_generate=True,
    logging_steps=1,
    report_to="wandb",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_en_vi["train"],
    eval_dataset=tokenized_en_vi["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.predict(tokenized_en_vi["test"])

