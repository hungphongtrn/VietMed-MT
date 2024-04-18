from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
import os
import fire
import torch
import gc
import logging
import yaml


metric = evaluate.load("sacrebleu")


def run_eval(MODEL, source_lang, target_lang):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    def preprocess_function_1(examples):
        inputs = [f"{source_lang}: {example[source_lang]}" for example in examples["translation"]]
        targets = [f"{target_lang}: {example[target_lang]}" for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    def preprocess_function_2(examples):
        inputs = [example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    if model.config.architectures[0] == "MBartForConditionalGeneration":
        preprocess_function = preprocess_function_2
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="en_XX", tgt_lang="vi_VN")
    elif model.config.architectures[0] == "T5ForConditionalGeneration" or "MT5ForConditionalGeneration":
        preprocess_function = preprocess_function_1
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # train_ds = load_dataset(train_ds_name)['train']
    # val_ds = load_dataset(train_ds_name)['dev']
    conv_test_ds = load_dataset("quan246/conv_test")['test']
    doc_test_ds = load_dataset("quan246/doc_test")['test']
    news_test_ds = load_dataset("quan246/news_test")['test']

    # tokennized_train = train_ds.map(preprocess_function, batched=True)
    # tokennized_val = val_ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # output_dir = f"{source_lang}_{target_lang}_{MODEL.split('/')[1]}_{train_ds_name.split('/')[1]}"
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     weight_decay=0.01,
    #     num_train_epochs=20,
    #     predict_with_generate=False,
    #     push_to_hub=True,
    #     report_to="wandb",
    #     gradient_accumulation_steps=16,
    #     run_name=output_dir,
    #     logging_steps=1,
    #     load_best_model_at_end=True,
    #     save_strategy="epoch",
    # )
    trainer = Seq2SeqTrainer(
        model=model,
        # args=training_args,
        # train_dataset=tokennized_train,
        # eval_dataset=tokennized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer.eval()

    tokenized_conv_test = conv_test_ds.map(preprocess_function, batched=True)
    tokenized_doc_test = doc_test_ds.map(preprocess_function, batched=True)
    tokenized_news_test = news_test_ds.map(preprocess_function, batched=True)

    conv_test_output = trainer.predict(tokenized_conv_test).metrics['test_bleu']
    doc_test_output = trainer.predict(tokenized_doc_test).metrics['test_bleu']
    news_test_output = trainer.predict(tokenized_news_test).metrics['test_bleu']

    # logging.info(f"Finished training {MODEL} from {source_lang} to {target_lang}")
    # logging.info(f"Results are saved in {output_dir}")
    results = {
        "_model": MODEL,
        "_direction": f"{source_lang} to {target_lang}",
        "conv_test": conv_test_output,
        "doc_test": doc_test_output,
        "news_test": news_test_output,
    }
    with open(f"./zero_shot_ret/{source_lang}_{target_lang}_{MODEL.split('/')[1]}.yaml", 'w') as f:
        yaml.dump(results, f)

    # cleanup
    torch.cuda.empty_cache()
    gc.collect()
    # return MODEL, source_lang, target_lang, conv_test_output, doc_test_output, news_test_output, model.num_parameters()


def run_with_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    MODEL = config['MODEL']
    source_lang = config['source_lang']
    target_lang = config['target_lang']
    run_eval(MODEL=MODEL, source_lang=source_lang, target_lang=target_lang)


if __name__ == "__main__":
    fire.Fire(run_with_config)