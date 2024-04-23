import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from hurst import compute_Hc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sklearn.linear_model


class JamesDataset(Dataset):
    """Dataset class for loading data from a text file and creating sliding window based segments."""
    def __init__(self, window_size=1024):
        super().__init__()
        # self.filename = "./papers/10.26434_chemrxiv-2022-djr5h.mmd"
        self.filename = "./james.txt"
        self.window_size = window_size
        self.words = self.load_and_tokenize_text()
        # self.words = " ".join(["jingle bells" * 4000])
        self.number_of_windows = len(self.words) - window_size + 1

    def load_and_tokenize_text(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                text = file.read()
            words = text.split()
            return words
        except IOError as e:
            print(f"Failed to read {self.filename}: {e}")
            return []

    def __len__(self):
        # return self.number_of_windows
        return 4000

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.number_of_windows:
            raise IndexError("Index out of bounds")
        word_segment = self.words[idx:idx + self.window_size]
        return ' '.join(word_segment)

DEFAULT_SCALES = [25, 50, 75, 100, 200, 300]
DEFAULT_SAMPLES_PER_DOC = 100
DEFAULT_EPS = 5e-2

def power_law_fit(x, y):
  """Fit a power law to the data (x, y)."""
  ones_logx = np.stack([np.ones_like(x), np.log(x)], axis=1)
  logy = np.log(y)
  clf = sklearn.linear_model.LinearRegression(
      fit_intercept=False).fit(ones_logx, logy)
  return clf.coef_[0], clf.coef_[1]  # coeff, exponent

def get_hurst_exponent(x,
                       scales=DEFAULT_SCALES,
                       samples_per_doc=DEFAULT_SAMPLES_PER_DOC):
  """Calculate the Hurst exponent.

  Args:
    x: increment process, 1D or 2D array.
    scales: granuality levels. Choose this carefully.
      If it is too large, you will have very few measurements.
      If it is too small, your estimate of the power law relation will be
      unreliable.
    samples_per_doc: number of samples per document. Ideally, this should be
      small and the number of documents should be large.

  Returns:
    H: Hurst exponent.
    sr: rescaled range estimates.
  """
  if x.ndim == 1:
    x = x.reshape((1, -1))
  elif x.ndim > 2:
    raise ValueError('x.ndim must be 1 or 2.')

  # calculate the rescaled range
  sr = []
  for n in scales:
    som = 0
    count = 0
    for i in range(len(x)):  # to get a reliable etimate, many documents are needed
      for _ in range(samples_per_doc):
        offset = np.random.randint(0, len(x[i]) - n)
        y = x[i, offset: n + offset]
        y = y - np.mean(y)
        Y = np.cumsum(y)
        R = max(Y) - min(Y)
        S = np.std(y)
        som += R / S
        count += 1
    sr.append(som / count)
    
  # estimate Hurst exponent
  return power_law_fit(scales, sr)[1], sr

# Initialize the dataset and DataLoader
WINDOW_SIZE = 300
BATCH_SIZE=1
dataset_james = JamesDataset(window_size=WINDOW_SIZE)
dataloader_james = DataLoader(dataset_james, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Init the tensorboard
writer = SummaryWriter('runs/jamesbetter_analysis')


# MODELPATH = "HuggingFaceHç4/zephyr-7b-alpha"
MODELPATH = "Qwen/Qwen1.5-1.8B"
model = AutoModelForCausalLM.from_pretrained(MODELPATH).half()
tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
tokenizer.pad_token = tokenizer.eos_token

# Setup for multi-GPU usage
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)  # Simplified multi-GPU usage for inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_probability_and_perplexity(input_ids, model):
    """Calculate the log probability of the first token in each sequence."""
    with torch.no_grad():
        # input_to_model = input_ids[:, :-1]
        # print(tokenizer.decode(input_to_model[0]))
        # print(input_to_model.shape)
        # outputs = model(input_to_model)
        # shift_logits = outputs.logits
        # print(shift_logits.shape)
        # pred_id = input_ids[0][-1]
        # print(pred_id)
        # log_probabilities_base_e = F.log_softmax(shift_logits, dim=-1)[0] # Get the first prediction
        # next_token_log_prob = log_probabilities_base_e[-1][pred_id]
        # print(f"Next token log prob : {next_token_log_prob}")

        # print(tokenizer.decode(pred_id.item()))
        # print(tokenizer.decode(torch.argmax(shift_logits[0][-1])))
        # return next_token_log_prob.item()
        input_to_model = input_ids[:, :-1]
        outputs = model(input_to_model)
        shift_logits = outputs.logits
        pred_id = input_ids[0][-1]
        log_probabilities_base_e = F.log_softmax(shift_logits, dim=-1) # Get the first prediction
        # print(tokenizer.decode(pred_id.item()))
        input_text = tokenizer.decode(input_to_model[0])
        # # print(f"input text: {input_text}")
        output_text = tokenizer.decode(shift_logits[0, :].argmax(dim=-1))

        # print(shift_logits.shape)
        # print(input_to_model.shape)
        # prob_of_correct = torch.gather(log_probabilities_base_e[:, :-1], 2, input_to_model[:, 1:].unsqueeze(-1))
        # prob_of_pred = torch.gather(log_probabilities_base_e[:, :-1], 2, log_probabilities_base_e[:, :-1].argmax(dim=-1).unsqueeze(-1))

        # print(prob_of_correct.shape)
        # print(f"output text: {output_text}")
        # for i, bob in enumerate(input_to_model[0, 1:]):
        #     print(tokenizer.decode(bob) + " " + tokenizer.decode(log_probabilities_base_e[0, i].argmax(dim=-1)) + " argmax: " + str(log_probabilities_base_e[0, i, log_probabilities_base_e[0, i].argmax(dim=-1).item()].item()) + " prob of correct:" + str(log_probabilities_base_e[0, i, bob].item()))
        # zipped = list(zip(input_text.split(), output_text.split(), list(prob_of_correct[0]), list(prob_of_pred[0])))
        # pp.pprint(zipped, width=200)
        # print(f"Predicted: {tokenizer.decode(log_probabilities_base_e[:, -1].argmax(dim=-1).item())},  {log_probabilities_base_e[0, -1, log_probabilities_base_e[:, -1].argmax(dim=-1).item()]}")
        # print(f"Should have been: {tokenizer.decode(pred_id)},  {log_probabilities_base_e[:, -1, pred_id].item()}")

        next_token_log_prob = log_probabilities_base_e[:, -1, pred_id].item()
        return next_token_log_prob

def log_metrics_and_plots(step, series, writer):
    """
    Log metrics and plots for the given series of log probabilities.
    
    Parameters:
        step (int): The current step or batch number in the processing loop.
        series (list): The series of log probabilities to analyze.
        writer (SummaryWriter): The TensorBoard writer object for logging.
    """
    # Convert and normalize the series
    series = np.array(series) / np.log(2) * -1  # Convert to base 2 for your specific use case
    normalized_series = (series - np.mean(series)) / np.std(series)
    # integral_series_normalized = np.cumsum(normalized_series)
    
    # Compute the Hurst parameter and related metrics
    # H, _, _ = compute_Hc(normalized_series, kind='change')
    H = get_hurst_exponent(normalized_series)[0]
    
    # Log the Hurst parameter and 'c' value to TensorBoard
    writer.add_scalar('Hurst/H', H, step)
    # writer.add_scalar('Hurst/c', c, step)

    # # Create plots
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # # Hurst equation plot
    # axs[0, 0].plot(data[0], c*data[0]**H, color="deepskyblue", label=f'H={H:.4f}, c={c:.4f}')
    # axs[0, 0].scatter(data[0], data[1], color="purple")
    # axs[0, 0].set_xscale('log')
    # axs[0, 0].set_yscale('log')
    # axs[0, 0].set_title("Hurst Equation")
    # axs[0, 0].set_xlabel('Time interval')
    # axs[0, 0].set_ylabel('R/S ratio')
    # axs[0, 0].legend()
    # axs[0, 0].grid(True)

    # # # Integral Series Normalized plot
    # # axs[0, 1].plot(integral_series_normalized, color="deepskyblue")
    # # axs[0, 1].set_title("Integral Series Normalized")
    # # axs[0, 1].set_xlabel('Time interval')
    # # axs[0, 1].grid(True)

    # # Next Token Probabilities Normalized plot
    # axs[1, 0].plot(normalized_series, color="deepskyblue")
    # axs[1, 0].set_title("Next Token Probs Normalized")
    # axs[1, 0].set_xlabel('Time interval')
    # axs[1, 0].grid(True)

    # # Optionally, you can leave one subplot empty or use it for another plot
    # axs[1, 1].axis('off')

    # # Log the figure to TensorBoard and then close the figure to prevent memory leaks
    # # writer.add_figure('Analysis/Plot', fig, step)
    # plt.close(fig)

i = 0
results_james = []
for batch in tqdm(dataloader_james):
    for chunk in batch:
        i += 1
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            # Manually distribute data to GPUs (if more than one)
        if torch.cuda.device_count() > 1:
            inputs = {key: val.to(device) for key, val in inputs.items()}
        else:
            inputs = inputs.to(device)

        log_prob_first_token = calculate_probability_and_perplexity(inputs.input_ids, model)
        results_james.append(log_prob_first_token)
        print(len(results_james))
        if i > 300 and i % 50 == 0:  # Save progress for every 10 batches
            log_metrics_and_plots(i, results_james, writer)
        # if i > 150:
writer.close()