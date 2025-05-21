from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from keys import API_KEYS
from tqdm import tqdm
from huggingface_hub import login
import numpy, argparse, logging, os, json, torch

HF_TOKEN = API_KEYS["hf_token"]        

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.data.append(record['document'].lower())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


class JsonlGraph(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                samp = record['sub'] + ' ' + record['rel'] + ' ' + record['obj']
                self.data.append(samp.lower())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


def save_jsonl(data, path):
    with open(path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    return


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def train(
    domain: str,
    model_name: str,
    *,
    learning_rate: float = 5e-5,
    max_length: int = 400,
    batch_size: int = 32,
    num_epochs: int = 30,
):
    """
    Fine-tune or evaluate a model to mine insights.

    Args:
        domain         : e.g. "AAN" or "OC"
        model_name     : HF model repo or local checkpoint
        learning_rate  : Learning rate
        max_length     : Truncation / padding length for the tokenizer
        batch_size     : Mini-batch size
        num_epochs     : Training epochs
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    # Configure LoRA for efficient fine-tuning
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules = target_modules,
        inference_mode=False,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05 
    )

    model = get_peft_model(model, lora_config)

    file_path_graph = f'../data/{domain}_graph.jsonl'
    file_path_doc = f'../data/{domain}_docs.jsonl'


    data_doc = JsonlDataset(file_path_doc, tokenizer, max_length)
    data_graph = JsonlGraph(file_path_graph, tokenizer, max_length)
    dataset = data_doc + data_graph


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    loss_epoch = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for input_ids, attention_mask in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.mean() 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


        Loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")
        loss_epoch += [{'epoch':epoch + 1, 'loss':Loss}]


    check_path(f'output/{domain}/')
    model_name = model_name.split('/')[-1]

    save_jsonl(loss_epoch, f'output/{domain}/loss.jsonl')
    path_model = f'output/{domain}/{model_name}'
    model.module.save_pretrained(path_model)
    tokenizer.save_pretrained(path_model)
    logger.info("Finished!")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Insight Miner training loop")
    p.add_argument("--domain",        required=True,  help="Target domain")
    p.add_argument("--model_name",    required=True,  help="HF model name or path")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length",    type=int, default=400)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_epochs",    type=int, default=30)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    login(HF_TOKEN)


    logger.info(
        "Starting training with parameters:\n"
        "  domain        = %s\n"
        "  model_name    = %s\n"
        "  learning_rate = %s\n"
        "  max_length    = %d\n"
        "  batch_size    = %d\n"
        "  num_epochs    = %d",
        args.domain,
        args.model_name,
        args.learning_rate,
        args.max_length,
        args.batch_size,
        args.num_epochs,
    )


    train(
        domain=args.domain,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

if __name__ == "__main__":
    main()



