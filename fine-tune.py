from datasets import load_dataset
from load_gxsum import load_gxsum
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
import json
import os
from tqdm import tqdm

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    "reddit_tifu": ("documents", "tldr"),
    "mediasum": ("document", "summary"),
}

def average(list):
    return sum(list)/len(list)

def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v, 4) for k, v in rouge_result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_result["gen_len"] = np.mean(prediction_lens)

    return rouge_result

def tokenize_func(examples: dict, tok: AutoTokenizer, col_name=None):
    model_input = tok(examples[col_name[0]], truncation=True, max_length=512, padding="max_length")
    label = tok(text_target=examples[col_name[1]], truncation=True, max_length=128, padding="max_length")
    model_input["labels"] = label["input_ids"]
    return model_input

def train():
    # ckpt = "facebook/bart-large"
    # Paper Table VI / IX: start from the checkpoint already fine-tuned on XSum.
    # For the "from scratch" ablation (LLM_Teached_*_FS), use google/pegasus-large instead.
    ckpt = "google/pegasus-xsum"
    save_dir = "Models/LLM_Teached_Pegasus"
    # GXSum configs: "small" (20k), "medium" (50k), "large" (90k). load_gxsum joins the documents back in.
    dataset = load_gxsum("small")

    # ds_train_dv = dataset["train"].train_test_split(test_size=0.1)
    # ds_dev = ds_train_dv["test"].train_test_split(test_size=0.5)
    # dataset = DatasetDict({
    #     "train": ds_train_dv["train"],
    #     "validation": ds_dev["train"],
    #     "test": ds_dev["test"]
    # })

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    tokenized_dataset = dataset.map(lambda x: tokenize_func(x, tokenizer), batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=ckpt)
    rouge = evaluate.load("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=16,
        predict_with_generate=True,
        generation_num_beams=6,
        generation_max_length=62,
        fp16=True,
        push_to_hub=True,
        gradient_accumulation_steps=4,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, rouge),
    )
    train_res = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_res.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.push_to_hub()

def test():
    ckpt = "vickt/LLM_Teached_BRIO_XSUM"
    model_name = ckpt.split("/")[1] if "/" in ckpt else ckpt
    save_dir = f"Models/{model_name}/reddit_test"
    dataset = load_dataset('json', data_files='datasets/reddit-tifu/reddit_tifu_test.json', split='train')
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    dataset_name = 'reddit_tifu'  # Change this to the dataset you are using
    col_name = summarization_name_mapping[dataset_name]
    tokenized_dataset = dataset.map(lambda x: tokenize_func(x, tokenizer, col_name), batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=ckpt)
    rouge = evaluate.load("rouge")
    # bert_score = evaluate.load("bertscore")

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to=None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, rouge),
    )
    predict_results = trainer.predict(tokenized_dataset, metric_key_prefix="predict", early_stopping=True, num_beams=8, max_length=100, min_length=10, length_penalty=0.8)
    metrics = predict_results.metrics
    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            res = []
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            for pred, src in zip(predictions, dataset[col_name[0]]):
                res.append({ "document": src, "summary": pred})
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
            with open(output_prediction_file, "w") as writer:
                json.dump(res, writer)
    metrics["predict_samples"] = len(tokenized_dataset)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    # trainer.push_to_hub()

def test_with_gen():
    ckpt = "Models/LLM_Teached_Bart"
    save_dir = "Models/LLM_Teached_Bart/wikihow_test"
    dataset = load_dataset('json', data_files='datasets/wikihow/wikihow_test_processed.json', split='train')
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    # model.cuda()  # superseded by device_map="auto"

    tokenized_dataset = dataset.map(lambda x: tokenize_func(x, tokenizer), batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=ckpt)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        collate_fn=data_collator,
        batch_size=32,
        shuffle=False,
    )

    sample = [tokenized_dataset[i] for i in range(4)]
    collated = data_collator(sample)
    for k, v in collated.items():
        print(k, type(v), v.shape if hasattr(v, "shape") else len(v))


    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")

    all_preds = []
    all_labels = []
    all_inputs = []
    device = model.device
    for batch in tqdm(dataloader, desc="Generating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=500,
                min_length=2,
                num_beams=8,
                length_penalty=0.8,
            )
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_inputs.extend(input_ids.cpu().numpy())

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v, 4) for k, v in rouge_result.items()}

    bert_score_result = bert_score.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    del bert_score_result['hashcode']
    bert_score_result = {k: round(sum(v)/len(v), 4) for k, v in bert_score_result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_result["gen_len"] = np.mean(prediction_lens)

    metrics = {**rouge_result, **bert_score_result}
    metrics["predict_samples"] = len(tokenized_dataset)

    os.makedirs(save_dir, exist_ok=True)
    res = []
    for pred, src in zip(decoded_preds, dataset["document"]):
        res.append({ "document": src, "summary": pred.strip() })
    output_prediction_file = os.path.join(save_dir, "generated_predictions.json")
    with open(output_prediction_file, "w") as writer:
        json.dump(res, writer, ensure_ascii=False, indent=2)

    # Save metrics
    metrics_file = os.path.join(save_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Inference done. Metrics:")
    print(json.dumps(metrics, indent=2))



if __name__ == "__main__":
    test()
    # train()
