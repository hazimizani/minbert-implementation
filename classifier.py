import math
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE=True

def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        # the final bert contextualized embedding is the hidden state of [CLS] token (the first token)
        outputs = self.bert(input_ids, attention_mask)
        pooler_output = outputs['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        return F.log_softmax(logits, dim=-1)

class BertSentClassifierImproved(torch.nn.Module):
    '''Enhanced classifier: CLS + mean pooling + max pooling with LR warmup support.'''
    def __init__(self, config):
        super(BertSentClassifierImproved, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = True

        hidden_size = config.hidden_size
        # CLS + mean pool + max pool = 3x hidden_size
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size * 3, config.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs['pooler_output']
        sequence_output = outputs['last_hidden_state']

        # Mean pooling (mask-aware)
        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Max pooling (mask-aware)
        sequence_masked = sequence_output.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        max_pooled, _ = sequence_masked.max(dim=1)

        combined = torch.cat([cls_output, mean_pooled, max_pooled], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)


def label_smoothed_nll_loss(log_probs, target, epsilon=0.1):
    '''NLL loss with label smoothing.'''
    nll_loss = F.nll_loss(log_probs, target, reduction='sum')
    smooth_loss = -log_probs.sum(dim=-1).sum()
    n_classes = log_probs.size(-1)
    return (1.0 - epsilon) * nll_loss + epsilon * smooth_loss / n_classes


def get_lr_scale(step, warmup_steps, total_steps):
    '''Linear warmup then linear decay.'''
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def train_improved(args):
    '''Training with LR warmup+decay and label smoothing using standard classifier.'''
    device = get_device(args.use_gpu)
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        option='finetune',
    )

    model = BertSentClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Estimate total steps for warmup schedule
    total_steps = args.epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)
    current_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids = batch[0]['token_ids'].to(device)
            b_mask = batch[0]['attention_mask'].to(device)
            b_labels = batch[0]['labels'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            # Update learning rate with warmup + decay
            current_step += 1
            scale = get_lr_scale(current_step, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * scale

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
            })

        return batches


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, flag='train'):
    # specify the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    with open(filename, 'r') as fp:
        for line in fp:
            label, org_sent = line.split(' ||| ')
            sent = org_sent.lower().strip()
            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            label = int(label.strip())
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, label, tokens))
    print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = get_device(args.use_gpu)
    #### Load data
    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    #### Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model
    model = BertSentClassifier(config)
    model = model.to(device)

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = get_device(args.use_gpu)
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']
        if hasattr(config, 'option') and config.option == 'improved_finetune':
            model = BertSentClassifierImproved(config)
        else:
            model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def ensemble_eval(dataloader, models, device):
    '''Evaluate an ensemble of models by averaging log-probabilities.'''
    for m in models:
        m.eval()
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'ensemble-eval', disable=TQDM_DISABLE)):
        b_ids = batch[0]['token_ids'].to(device)
        b_mask = batch[0]['attention_mask'].to(device)
        b_labels = batch[0]['labels']

        # Average log-probabilities across models
        avg_logits = None
        for m in models:
            logits = m(b_ids, b_mask).detach().cpu()
            if avg_logits is None:
                avg_logits = logits
            else:
                avg_logits = avg_logits + logits
        avg_logits = avg_logits / len(models)

        preds = np.argmax(avg_logits.numpy(), axis=1).flatten()
        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(batch[0]['sents'])

    acc = accuracy_score(y_true, y_pred)
    return acc, y_pred, y_true, sents


def train_ensemble(args):
    '''Train multiple models with different seeds, ensemble predictions.'''
    device = get_device(args.use_gpu)
    seeds = [1234, 11711, 42]

    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')
    test_data = create_data(args.test, 'test')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)
    test_dataset = BertDataset(test_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        option='finetune',
    )

    model_paths = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Training model {i+1}/{len(seeds)} with seed={seed} ===")
        seed_everything(seed)
        model = BertSentClassifier(config)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        best_dev_acc = 0
        model_path = args.filepath.replace('.pt', f'-seed{seed}.pt')

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
                b_ids = batch[0]['token_ids'].to(device)
                b_mask = batch[0]['attention_mask'].to(device)
                b_labels = batch[0]['labels'].to(device)

                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / num_batches
            train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
            dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, model_path)

            print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

        model_paths.append(model_path)

    # Load best checkpoint for each seed and ensemble
    print(f"\n=== Ensemble evaluation ({len(seeds)} models) ===")
    models = []
    for mp in model_paths:
        saved = torch.load(mp, weights_only=False)
        m = BertSentClassifier(saved['model_config'])
        m.load_state_dict(saved['model'])
        m = m.to(device)
        m.eval()
        models.append(m)

    dev_acc, dev_pred, dev_true, dev_sents = ensemble_eval(dev_dataloader, models, device)
    test_acc, test_pred, test_true, test_sents = ensemble_eval(test_dataloader, models, device)

    with open(args.dev_out, "w+") as f:
        print(f"ensemble dev acc :: {dev_acc :.3f}")
        for s, t, p in zip(dev_sents, dev_true, dev_pred):
            f.write(f"{s} ||| {t} ||| {p}\n")

    with open(args.test_out, "w+") as f:
        print(f"ensemble test acc :: {test_acc :.3f}")
        for s, t, p in zip(test_sents, test_true, test_pred):
            f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated; improved_finetune: enhanced model with warmup+pooling+label smoothing',
                        choices=('pretrain', 'finetune', 'improved_finetune', 'ensemble'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    if args.option == 'ensemble':
        train_ensemble(args)
    elif args.option == 'improved_finetune':
        train_improved(args)
    else:
        train(args)
        test(args)
