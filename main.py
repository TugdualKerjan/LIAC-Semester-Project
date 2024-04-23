from transformers import AutoModelForCausalLM, AutoTokenizer
from paperDatabase import TextDataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import pprint as pp

device = "cuda" # the device to load the model onto

models_path = {"qwen" : "Qwen/Qwen1.5-7B-Chat", "mistral-inst": "mistralai/Mistral-7B-Instruct-v0.1",  "mistral": "mistralai/Mistral-7B-v0.1", "zephyr": "HuggingFaceH4/zephyr-7b-alpha"}

MODELPATH = models_path["mistral-inst"]

model = AutoModelForCausalLM.from_pretrained(MODELPATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
tokenizer.pad_token = tokenizer.eos_token 

batch_size = 4

dataset = TextDataset("./papers")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

results = []

def save_results(results):
    with open("./qalong/10-Papers-Qwen1.5-7B-Chat.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=4)

for batch in tqdm(dataloader):
    for batch_text in batch:
        messages = [{"role": "user", "content":f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia, of about the same length as the inputed text. Transform the latex mentionning molecules into wikipedia style: {batch_text}"""}]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=2500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pp.pprint("Input text: " + batch_text + ", \nOutput text: " + response)
        results.append({"input_text": batch_text, "output_text": response})

        # Save periodically or based on a condition, for example, every 10 batches
        # if len(results) % 10 == 1:
            # save_results(results)

save_results(results)