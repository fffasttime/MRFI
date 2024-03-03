import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

# or  text_transform = XLMR_BASE_ENCODER.transform()

from torch.utils.data import DataLoader

from torchtext.datasets import SST2

batch_size = 64

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")


# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    return text_transform(x[0]), x[1]


train_datapipe = train_datapipe.map(apply_transform)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(apply_transform)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)


num_classes = 2
input_dim = 768
TRAIN_MODEL = False 


from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
if TRAIN_MODEL:
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
else:
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head, load_weights=False)
    model.load_state_dict(torch.load('roberta_sst2.pt'))
model.to(DEVICE)

import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()


def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    #model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions

from tqdm import tqdm

if TRAIN_MODEL:
    
    num_epochs = 2

    for e in range(num_epochs):
        for batch in tqdm(train_dataloader):
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            train_step(input, target)

        loss, accuracy = evaluate()
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
        torch.save(model.state_dict(), 'roberta_sst2.pt')
else:
    loss, accuracy = evaluate()
    print("loss = [{}], accuracy = [{}]".format(loss, accuracy))
    from mrfi import MRFI, EasyConfig
    econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
    econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
    econfig.faultinject[0]['quantization']['scale_factor'] = 1
    econfig.faultinject[0]['selector']['rate'] = 1.6e-4
    econfig.set_module_used(0, module_name = ['linear', 'proj'])
    fi_model = MRFI(model, econfig)
    loss, accuracy = evaluate()
    print("loss = [{}], accuracy = [{}]".format(loss, accuracy))
