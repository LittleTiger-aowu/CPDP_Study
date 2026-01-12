from __future__ import absolute_import, division, print_function
import csv
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from prefetch_generator import BackgroundGenerator
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from itertools import cycle
from imblearn.over_sampling import RandomOverSampler
from model import Model
import MMD as MMD
logger = logging.getLogger(__name__)
from parser.DFG import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript

from parser.utils import (remove_comments_and_docstrings,
                          tree_to_token_index,
                          index_to_code_token,
                          tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function = {
    'java': DFG_java,
}


# load parsers
parsers = {}
parsers['java'] = Parser()
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 source_ids,
                 position_idx,
                 attn_mask,
                 label,
                 idx,
                 subtrees
                 ):
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.attn_mask = attn_mask
        self.label = label
        self.idx = idx
        self.subtrees = subtrees


def load_dataset(project_name, args):
    with open(os.path.join(args.file_dir, project_name) + ".pkl", 'rb') as f:
        data = pickle.load(f)

    source_ids = torch.tensor([f.source_ids for f in data])
    position_idx = torch.tensor([f.position_idx for f in data])
    attn_mask = [f.attn_mask.unsqueeze(0) for f in data]
    attn_mask = torch.cat(attn_mask, dim=0)
    label = torch.tensor([f.label for f in data])
    idx = torch.tensor([f.idx for f in data])
    subtrees = [f.subtrees for f in data]

    return source_ids, position_idx, attn_mask, label, idx, subtrees


def train(args, train_dataset, train_AST, test_dataset, test_AST, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1,
                                  drop_last=True, pin_memory=True)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, num_workers=1)
    test_dataloader = cycle(test_dataloader)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 1 * 100
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)
    EarlyStopping = False
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.99))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0

    model.zero_grad()

    for idx in range(args.epochs):

        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0

        for step, batch in enumerate(bar):
            model.train()
            bar_test = next(test_dataloader)
            (target_ids, target_position_idx, target_attn_mask, target_label, target_idx) = tuple(
                t.to(device) for t in bar_test)
            (source_ids, source_position_idx, source_attn_mask, source_label, source_idx) = [x.to(args.device) for x in
                                                                                             batch]

            loss, logits, Train_AST, Test_AST, Train_AST_hn, Train_AST_cn, Test_AST_hn, Test_AST_cn = model(
                source_ids, source_position_idx, source_attn_mask, source_label, source_idx, train_AST,
                target_ids, target_position_idx, target_attn_mask, target_label, target_idx, test_AST)

            loss_mmd_AST_hn = MMD.mmd_loss(Train_AST_hn, Test_AST_hn, 5)
            loss_mmd_AST_cn = MMD.mmd_loss(Train_AST_cn, Test_AST_cn, 5)
            loss = loss + (loss_mmd_AST_hn + loss_mmd_AST_cn) * 2

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            if avg_loss < args.EarlyStopping_Loss and tr_num >= 10:
                EarlyStopping = True
                break
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
        if EarlyStopping:
            torch.cuda.empty_cache()
            break
    checkpoint_prefix = 'checkpoint-best-f1'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, args.train_data_file + 'module') else model
    output_dir = os.path.join(output_dir, '{}'.format(args.train_data_file + 'model.bin'))
    torch.save(model_to_save.state_dict(), output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)


def test(args, test_dataset, target_subtrees, model):

    checkpoint_prefix = 'checkpoint-best-f1/' + args.train_data_file + 'model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir), False)
    model.to(args.device)

    eval_sampler = RandomSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.test_batch_size, num_workers=1,
                                 pin_memory=True)
    # multi-gpu evaluate
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info(args.train_data_file + '--' + args.test_data_file)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.train()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (target_ids, target_position_idx, target_attn_mask, target_label, target_idx) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(target_ids, target_position_idx, target_attn_mask, target_label, target_idx, target_subtrees)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(target_label.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    y_preds = logits[:] > best_threshold

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_trues, logits)
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)
    result = {
        "train": 0,
        'test': 0,
        "eval_f1": float(f1),
        "eval_acc": float(acc),
        "eval_auc": float(auc),
        "eval_mcc": float(mcc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_precision": float(precision),
        "eval_threshold": best_threshold,
    }
    logger.info("***** Test results *****")
    logger.info(args.train_data_file + '--' + args.test_data_file)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result


def main(source_project, target_project):
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=source_project, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--test_data_file", default=target_project, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--file_dir", default="dataset/PROMISE", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_name_or_path", default='graphcodebert-base', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="graphcodebert-base/config.json", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=None, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=None, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_snippets", default=6, type=int,
                        help="The maximum length of BERT will limit the length of code that can be processed, and here "
                             "we recommend dividing long code into code snippets of a processable length.")
    parser.add_argument("--do_train", default=None, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=None, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=None, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--test_batch_size", default=None, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=None, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=None, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=None, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=None, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=None, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--EarlyStopping_Loss", default=None, type=float,
                        help="Considering setting an Earlystopping loss to avoid overfitting.")
    parser.add_argument('--seed', type=int, default=None,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=None,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                           output_hidden_states=True)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = Model(model, config, tokenizer, args)

    source_ids, source_position_idx, source_attn_mask, source_label, source_idx, source_subtrees = load_dataset(args.train_data_file, args)
    target_ids, target_position_idx, target_attn_mask, target_label, target_idx, target_subtrees = load_dataset(args.test_data_file, args)
    randover = RandomOverSampler()
    sample, label = randover.fit_resample(X=source_idx.unsqueeze(1), y=source_label)
    temp = []
    for i in sample:
        if i[0] not in temp:
            temp.append(i[0])
        else:
            source_ids = torch.cat((source_ids, source_ids[i[0]].unsqueeze(0)), dim=0)
            source_position_idx = torch.cat((source_position_idx, source_position_idx[i[0]].unsqueeze(0)), dim=0)
            source_attn_mask = torch.cat((source_attn_mask, source_attn_mask[i[0]].unsqueeze(0)), dim=0)
            source_label = torch.cat((source_label, source_label[i[0]].unsqueeze(0)), dim=0)
    sample = torch.tensor(sample.squeeze(1))

    train_dataset = torch.utils.data.TensorDataset(source_ids, source_position_idx, source_attn_mask, source_label, sample)
    test_dataset = torch.utils.data.TensorDataset(target_ids, target_position_idx, target_attn_mask, target_label, target_idx)

    train(args, train_dataset, source_subtrees, test_dataset, target_subtrees, model)
    result = test(args, test_dataset, target_subtrees, model)
    return result

if __name__ == "__main__":
    CPDP = [
        ['ant-1.7',
         ['camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['camel-1.6',
         ['ant-1.7', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['forrest-0.8',
         ['ant-1.7', 'camel-1.6', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['ivy-2.0',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['jEdit-4.2',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'log4j-1.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['log4j-1.2',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'lucene-2.4', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['lucene-2.4',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'poi-3.0', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['poi-3.0',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'synapse-1.2',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['synapse-1.2',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0',
          'velocity-1.6.1',
          'xalan-2.7', 'xerces-1.4.4']],
        ['velocity-1.6.1',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0',
          'synapse-1.2',
          'xalan-2.7', 'xerces-1.4.4']],
        ['xalan-2.7',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0',
          'synapse-1.2',
          'velocity-1.6.1', 'xerces-1.4.4']],
        ['xerces-1.4.4',
         ['ant-1.7', 'camel-1.6', 'forrest-0.8', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0',
          'synapse-1.2',
          'velocity-1.6.1', 'xalan-2.7']]
    ]
    for projects in CPDP:
        source = projects[0]
        for target in projects[1]:
            result = main(source_project=source, target_project=target)
            result['train'] = source
            result['test'] = target
            df = pd.DataFrame(
                columns=['train', 'test', 'eval_f1', 'eval_acc', 'eval_auc', 'eval_mcc', 'eval_precision',
                         'eval_recall', 'lr',
                         'epoch', ]).from_dict(data=result, orient='index').T

            save_path = 'CPDP' + '.csv'
            # 判断文件是否存在
            if os.path.exists(save_path):
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(save_path, mode='w', index=False)
