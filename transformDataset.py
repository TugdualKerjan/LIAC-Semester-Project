import datasets
import torch
import json
import transformers
import pprint as pp
import numpy as np
from tqdm import tqdm

np.random.seed(2024)

model_name = "HuggingFaceH4/zephyr-7b-alpha"
save_path = './processed_qa_scientific_papers-zephyr2.json'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

model = transformers.AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

ds = datasets.load_dataset('scientific_papers', 'arxiv', split='validation')

MAX_LENGTH = 512

processed_data = []

for article in tqdm(ds.select(range(10))):
    processed_article = []
    
    sentences = article['article'].split("\n")
    paragraphs = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    # Processing each chunk
    for chunk_index, chunk in enumerate(paragraphs):
        if chunk_index == 0:
            prompt = [
                {"role": "system", "content": f"You produce the next paragraph in high quality wikipedia-like English based on the content the user provides. You change any formatting into Wikipedia style. You rephrase, don't summarize. Do not start with an introductory phrase, simply start writing the content, which should simply be a rephrased version of what the user provides."},{"role": "user", "content": chunk}
            ]
        else:
            # Use the last paragraph processed as context for continuity
            context = processed_article[-1]  # Using the most recent paragraph as context
            prompt = [
                 {"role": "system", "content": f"You produce the next paragraph in high quality wikipedia-like English based on the content the user provides. You change any formatting into Wikipedia style. You rephrase, don't summarize. Do not start with an introductory phrase, simply start writing the content, which should simply be a rephrased version of what the user provides. Here is the previously rephrased content which you should base yourself on but not repeat: {context}"},{"role": "user", "content": chunk}
            ]

        # Generate response from the model
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=MAX_LENGTH,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pp.pprint(response)
        processed_article.append(response)

    # Store processed article
    processed_data.append({'article': " ".join(processed_article), 'abstract': article['abstract'], 'section_names': article['section_names']})

save_path = 'processed_scientific_papers_zepyhr2.json'


# Save results to a file
try:
    with open(save_path, 'w') as f:
        json.dump(processed_data, f)
    print(f"Processed dataset saved to {save_path}")
except IOError as e:
    print(f"Failed to save the file: {e}")
