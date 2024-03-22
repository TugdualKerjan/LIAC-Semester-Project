from torch.utils.data import Dataset
import os

BATCH_SIZE = 4

class TextDataset(Dataset):
    def __init__(self, markdown_folder):
        super().__init__()
        self.paragraphs = self.load_paragraphs(markdown_folder)

    def load_paragraphs(self, folder):
        paragraphs = []
        # Loop through each file in the markdown folder
        for filename in os.listdir(folder):
            if filename.endswith('.mmd'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Split the text into paragraphs based on two newline characters
                    file_paragraphs = text.split('\n\n')
                    file_paragraphs_without_titles = [ x for x in file_paragraphs if "##" not in x and "**" not in x and "MISSING_PAGE_FAIL" not in x]
                    paragraphs.extend(file_paragraphs_without_titles)
                    paragraphs.append(text)
        return paragraphs

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]

