from torch.utils.data import Dataset
import os

BATCH_SIZE = 4

class TextDataset(Dataset):
    def __init__(self, markdown_folder):
        super().__init__()
        self.paragraphs = self.load_paragraphs(markdown_folder)

    def load_paragraphs(self, folder):
        chunks = []
        # Loop through each file in the markdown folder
        for filename in os.listdir(folder):
            if filename.endswith('.mmd'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    words = text.split()
                    chunks.extend([" ".join(words[i:i+2100]) for i in range(0, len(words), 2100)])
                    # Split the text into paragraphs based on two newline characters
                    # file_paragraphs = text.split('\n\n')
                    # file_paragraphs_without_titles = [ x for x in text if "##" not in x and "**" not in x and "MISSING_PAGE_FAIL" not in x]
                    # chunks.extend(text)
                    # paragraphs.append(text)
        return chunks

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]

