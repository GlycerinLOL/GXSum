import os
import sys
import openai
import pandas as pd
import json
from openai_multi_client import OpenAIMultiOrderedClient
import datetime
from tqdm import tqdm

# Set your API key
# data_dir = "xsum_train_20000"
data_dir = "datasets/samsum/test.csv"
save_dir = "GPT_Eval_result/samsum/gpt_eval(samsum).json"
openai.api_key = os.environ["OPENAI_API_KEY"]
dataset = pd.read_csv(data_dir)
model = "gpt-4-1106-preview"
# model = "gpt-3.5-turbo"

def make_requests(api, dataset):
    count = 0
    Dimension = ["Coherence", "Consistency", "Hallucination", "Relevance", "Informativeness", "Fluency"]
    for index, data in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        article = data['original_article']
        summary1 = data['Summary_1']
        # summary1 = data['Summary_2']
        # api.request(data={
        #     "response_format": { "type": "json_object" },
        #     "messages": [{
        #         "role": "user",
        #         "content": f"Evaluate the quality of summaries written for a news article. Rate each summary on six dimensions: {Dimension[0]}, {Dimension[1]}, {Dimension[2]}, {Dimension[3]}, {Dimension[4]} and {Dimension[5]}. You should rate on a scale from 1 (worst) to 5 (best), without any explanation. Please return by json format, with dimensions as key and score as value. Article: {article} Summary1: {summary1}"}],
        #         "max_tokens": 256,
        #         "temperature": 0.0,
        #     }, metadata={'article': article, 'summary': summary1})
        api.request(data={
            "response_format": { "type": "json_object" },
            "messages": [{
                "role": "user",
                "content": f"Evaluate the quality of summaries written for a dialogue. Rate each summary on six dimensions: {Dimension[0]}, {Dimension[1]}, {Dimension[2]}, {Dimension[3]}, {Dimension[4]} and {Dimension[5]}. You should rate on a scale from 1 (worst) to 5 (best), without any explanation. Please return by json format, with dimensions as key and score as value. Dialogue: {article} Summary1: {summary1}"}],
                "max_tokens": 256,
                "temperature": 0.0,
            }, metadata={'article': article, 'summary': summary1})

        count += 1
        # if count == 20:
        #     break


def generate():
    start = datetime.datetime.now()
    ans = []
    failed = 0
    api = OpenAIMultiOrderedClient(concurrency = 30, endpoint = "chats", data_template={"model": model}, max_retries=30, wait_interval=0.5, retry_multiplier=2)
    api.run_request_function(make_requests, api, dataset)
    # api.run_request_function(make_requests, dataset)
    for result in api:
        if result.failed:
            failed += 1
            continue
        else:
            article = result.metadata['article']
            summary = result.metadata['summary']
            response = json.loads(result.response['choices'][0]['message']['content'])
            tmp = {"document": article, "summary": summary, "score": response}
            ans.append(tmp)
    with open(save_dir, "w") as f:
        json.dump(ans, f)
    print("failed: ", failed)
    end = datetime.datetime.now()
    process_time = end - start
    print(process_time)
    print('*' * 20)
    print(f"Total failed: {failed}/{len(dataset)}")
    print('*' * 20)
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    generate()
