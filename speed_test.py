import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset
import time
from tqdm import tqdm

def test_original_method():
    """Benchmark the speed of the original method."""
    print("Testing original method...")

    ckpt = "Models/LLM_Teached_Bart"
    dataset = load_dataset('json', data_files='datasets/wikihow/wikihow_test_processed.json', split='train')

    # Use only the first 50 samples for this benchmark
    test_dataset = dataset.select(range(min(50, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    # Mimic the original method's tokenization
    def tokenize_func(examples):
        model_input = tokenizer(examples["document"], truncation=True, max_length=1024)
        label = tokenizer(text_target=examples["summary"], truncation=True, max_length=512)
        model_input["labels"] = label["input_ids"]
        return model_input

    tokenized_dataset = test_dataset.map(tokenize_func, batched=True)

    start_time = time.time()

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_dataset), 8)):
            batch = tokenized_dataset[i:i+8]

            # Prepare inputs
            input_ids = torch.tensor(batch["input_ids"]).cuda()
            attention_mask = torch.tensor(batch["attention_mask"]).cuda()

            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=500,
                min_length=2,
                num_beams=8,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode
            summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(summaries)

    end_time = time.time()
    return end_time - start_time, len(results)

def test_pipeline_method():
    """Benchmark the speed of the HF pipeline method."""
    print("Testing pipeline method...")

    ckpt = "Models/LLM_Teached_Bart"
    dataset = load_dataset('json', data_files='datasets/wikihow/wikihow_test_processed.json', split='train')

    # Use only the first 50 samples for this benchmark
    test_dataset = dataset.select(range(min(50, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    # Build the pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=16
    )

    start_time = time.time()

    documents = test_dataset["document"]
    results = []

    for i in range(0, len(documents), 16):
        batch_docs = documents[i:i+16]
        summaries = summarizer(
            batch_docs,
            max_length=500,
            min_length=2,
            do_sample=False,
            num_beams=8,
            early_stopping=True
        )

        for summary in summaries:
            results.append(summary[0]['summary_text'])

    end_time = time.time()
    return end_time - start_time, len(results)

def test_direct_method():
    """Benchmark the speed of the direct-model method."""
    print("Testing direct method...")

    ckpt = "Models/LLM_Teached_Bart"
    dataset = load_dataset('json', data_files='datasets/wikihow/wikihow_test_processed.json', split='train')

    # Use only the first 50 samples for this benchmark
    test_dataset = dataset.select(range(min(50, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    start_time = time.time()

    documents = test_dataset["document"]
    results = []

    with torch.no_grad():
        for i in range(0, len(documents), 16):
            batch_docs = documents[i:i+16]

            # Tokenize
            inputs = tokenizer(
                batch_docs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_length=500,
                min_length=2,
                num_beams=8,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode
            summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(summaries)

    end_time = time.time()
    return end_time - start_time, len(results)

if __name__ == "__main__":
    print("Starting speed benchmark...")
    print("="*50)

    # Benchmark the original method
    try:
        original_time, original_count = test_original_method()
        print(f"Original method: {original_time:.2f}s, {original_count} samples")
    except Exception as e:
        print(f"Original method failed: {e}")
        original_time = float('inf')
        original_count = 0

    print()

    # Benchmark the pipeline method
    try:
        pipeline_time, pipeline_count = test_pipeline_method()
        print(f"Pipeline method: {pipeline_time:.2f}s, {pipeline_count} samples")
    except Exception as e:
        print(f"Pipeline method failed: {e}")
        pipeline_time = float('inf')
        pipeline_count = 0

    print()

    # Benchmark the direct method
    try:
        direct_time, direct_count = test_direct_method()
        print(f"Direct method: {direct_time:.2f}s, {direct_count} samples")
    except Exception as e:
        print(f"Direct method failed: {e}")
        direct_time = float('inf')
        direct_count = 0

    print()
    print("="*50)
    print("Speed comparison results:")
    print("="*50)

    methods = [
        ("Original method", original_time, original_count),
        ("Pipeline method", pipeline_time, pipeline_count),
        ("Direct method", direct_time, direct_count)
    ]

    # Drop any method that failed
    valid_methods = [(name, time, count) for name, time, count in methods if time != float('inf')]

    if valid_methods:
        # Sort by speed
        valid_methods.sort(key=lambda x: x[1])

        fastest = valid_methods[0]
        print(f"Fastest method: {fastest[0]}")
        print(f"Speed: {fastest[2]/fastest[1]:.2f} samples/sec")

        print("\nDetailed comparison:")
        for i, (name, time, count) in enumerate(valid_methods):
            speed = count/time if time > 0 else 0
            print(f"{i+1}. {name}: {time:.2f}s, {speed:.2f} samples/sec")

            if i > 0:
                speedup = valid_methods[0][1] / time
                print(f"   {speedup:.2f}x slower than the fastest method")
    else:
        print("All methods failed")
