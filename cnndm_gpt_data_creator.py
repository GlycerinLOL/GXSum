import os
import sys
import openai
from datasets import load_from_disk
import json
from openai_multi_client import OpenAIMultiOrderedClient
import datetime
from tqdm import tqdm
import tiktoken

# Set your API key
data_dir = "cnndm/cnndm_test_all"
openai.api_key = os.environ["OPENAI_API_KEY"]

dataset = load_from_disk(data_dir)
model = "gpt-4-1106-preview"
# model = "gpt-3.5-turbo"
output_file = "cnndm_test_gpt4_turbo.json"
def make_requests(api, dataset):
    enc = tiktoken.encoding_for_model(model)
    count = 0
    for data in tqdm(dataset):
        doc = data['article']
        ref = data['highlights']
        ref_len = len(enc.encode(ref))
        api.request(data={
            "messages": [{
                "role": "user",
                "content": f"Assuming you are an abstract writer, responsible for writing summaries of articles. Given the source article: {doc}, please write a summary between {ref_len - 5} to {ref_len + 5} words about this article. please ensure that the summary is grammatically correct and coherent."
            }]
        }, metadata={'doc': doc})
        count += 1
        # if count == 100:
        #     break


def generate():
    start = datetime.datetime.now()
    chunk_num = 1
    ans = []
    failed = 0
    for i in range(chunk_num):
        api = OpenAIMultiOrderedClient(concurrency = 30, endpoint = "chats", data_template={"model": model}, max_retries=70, wait_interval=0.5, retry_multiplier=2)
        shard = dataset.shard(num_shards=chunk_num, index=i)
        api.run_request_function(make_requests, api, shard)
    # api.run_request_function(make_requests, dataset)
        for result in api:
            if result.failed:

                failed += 1
                continue
            else:
                doc = result.metadata['doc']
                response = result.response['choices'][0]['message']['content']
                tmp = {"document": doc, "summary": response}
                ans.append(tmp)
        with open(output_file, "w") as f:
            json.dump(ans, f, indent = 3)
        print("saved with chunk num: ", i)
    end = datetime.datetime.now()
    process_time = end - start
    print(process_time)
    print('*' * 20)
    print(f"Total failed: {failed}/{len(dataset)}")
    print('*' * 20)
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    generate()
