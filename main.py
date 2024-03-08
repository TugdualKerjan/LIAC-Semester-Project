from transformers import AutoModelForCausalLM, AutoTokenizer
from paperDatabase import TextDataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import pprint as pp

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

batch_size = 4

dataset = TextDataset("./papers")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
prompt = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: For the following paragraph give me a paraphrase of the same using
a very small vocabulary and extremely simple sentences that a toddler will
understand:"""


results = []

def save_results(results):
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=4)

for batch in tqdm(dataloader):
    for paragraph_text in batch:
        pp.pprint("Input text: " + paragraph_text)

        messages = [{"role": "user", "content": prompt + paragraph_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=128, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pp.pprint("Output text: " + response)
        results.append({"input_text": paragraph_text, "output_text": response})

        # Save periodically or based on a condition, for example, every 10 batches
        if len(results) % 10 == 0:
            save_results(results)

save_results(results)