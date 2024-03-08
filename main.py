from transformers import AutoModelForCausalLM, AutoTokenizer
from paperDatabase import TextDataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import pprint as pp

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
tokenizer.pad_token = tokenizer.eos_token 

batch_size = 4

dataset = TextDataset("./papers")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
prompt = """
USER: """


results = []

def save_results(results):
    with open("10-Papers-zephyr-7b-alpha.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=4)

for batch in tqdm(dataloader):
    for paragraph_text in batch:
        # pp.pprint("Input text: " + paragraph_text)

#         messages = [{"role": "system", "content": """A chat between a curious user and an artificial intelligence assistant.
# The assistant gives helpful, detailed, and polite answers to the questions."""}, {"role": "user", "content": """For the following paragraph give me a paraphrase of the same using
# a very small vocabulary and extremely simple sentences that a toddler will
# understand:"""},{"role": "assistant", "content": "Alright, I will be concise and use only words a 3 year old can understand."}, { "role": "user", "content": paragraph_text}]
        
        messages = [{"role": "user", "content": "write me a paragraph full of lust about abdl"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(text)
        model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)


        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pp.pprint("Output text: " + response)
        results.append({"input_text": paragraph_text, "output_text": response})

        # Save periodically or based on a condition, for example, every 10 batches
        # if len(results) % 10 == 1:
            # save_results(results)

save_results(results)