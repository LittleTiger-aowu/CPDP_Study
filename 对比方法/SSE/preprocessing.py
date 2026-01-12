from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
import pandas as pd
from ParseTOASTPath import ParseToASTPath
from parser.DFG import DFG_java
import pickle
from parser.utils import (remove_comments_and_docstrings,
                          tree_to_token_index,
                          index_to_code_token, )
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

dfg_function = {
    'java': DFG_java,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parsers = {}
parsers['java'] = Parser()
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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


class Example(object):
    def __init__(self, idx, source, label):
        self.idx = idx
        self.source = source
        self.label = label


def append_suffix(df):
    for i in range(len(df['file_name'])):
        df.loc[i, 'file_name'] = df.loc[i, 'file_name'] + ".java"
    return df


def extract_handcraft_instances(path):
    handcraft_data = pd.read_csv(path)
    handcraft_data = append_suffix(handcraft_data)
    handcraft_instances = np.array(handcraft_data['file_name'])
    handcraft_label = np.array(handcraft_data['bug'])
    handcraft_instances = handcraft_instances.tolist()
    handcraft_label = handcraft_label.tolist()

    return handcraft_instances, handcraft_label


def ast_parse_CPDP(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        return content


def read_examples(args):
    examples = []
    filename = args.project_name
    root_path_source = "dataset/PROMISE/projects/"
    root_path_csv = "dataset/PROMISE/csvs/"
    path_train_source = root_path_source + filename
    path_train_handcraft = root_path_csv + filename + '.csv'
    handcraft_instances, handcraft_label = extract_handcraft_instances(path_train_handcraft)
    package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']
    name_to_label = dict(zip(handcraft_instances, handcraft_label))
    idx = 0
    for dir_path, dir_names, file_names in os.walk(path_train_source):

        if len(file_names) == 0:
            continue

        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue

        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')

        for file in file_names:
            if file.endswith('java'):
                if str(package_name + "." + str(file)) not in handcraft_instances:
                    continue

                label = name_to_label[str(package_name + "." + str(file))]
                if label != 0:
                    label = 1
                    # existed_file_names.append(str(package_name + "." + str(file)))
                with open(str(os.path.join(dir_path, file)), 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                # all_data.append((label, content, str(os.path.join(dir_path, file))))
                examples.append(Example(label=label, source=content, idx=idx))
                idx += 1
    return examples


def convert_examples_to_features(examples, tokenizer, args):
    features = []
    parser = parsers['java']
    for example_index, example in tqdm(enumerate(examples), total=len(examples)):
        code_tokens, dfg = extract_dataflow(example.source, parser, 'java')
        subtrees_tokens = []
        subtrees = ParseToASTPath(example.source)
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                       enumerate(code_tokens)]


        # extract data flow
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        i = -1
        flag = False
        for idx1, subtree in enumerate(subtrees):
            subtree_token = []
            for idx2, paths in enumerate(subtree):
                i += 1
                start, end = ori2cur_pos[i]
                if end <= args.code_snippets * (args.code_length - 2):
                    for j in range(start, end):
                        subtree_token.append(subtrees[idx1][idx2])
                else:
                    for j in range(start, args.code_snippets * (args.code_length - 2)):
                        subtree_token.append(subtrees[idx1][idx2])
                    flag = True
                    break
            subtrees_tokens.append(subtree_token)
            if flag:
                break

        # reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            try:
                dfg[idx] = x[:1] + tuple(ori2cur_pos[x[1]]) + x[2:-1] + (
                [reverse_index[i] for i in x[-1] if i in reverse_index],)
            except:
                break
        source_idss = []
        position_idxs = []
        code_tokens = [y for x in code_tokens for y in x]
        idx1 = 0
        idx2 = 0
        for k in range(args.code_snippets):
            # truncating
            code_token = code_tokens[k * (args.code_length - 2)
                                     : (k + 1) * (args.code_length - 2)][:args.code_length - 2]
            source_tokens = [tokenizer.cls_token] + code_token + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            cur_dfg = []
            while True:
                if len(dfg) > 0 and dfg[0][2] <= (k + 1) * (args.code_length - 2):
                    idx1 += 1
                    cur = dfg.pop(0)
                    comefrom = []
                    for ele in cur[-1]:
                        if ele > idx2 and ele - idx2 < args.data_flow_length:
                            comefrom.append(ele - idx2)

                    if cur[1] < k * (args.code_length - 2):
                        cur = tuple(cur[0:1]) + tuple([0]) + tuple([cur[2] - k * (args.code_length - 2)]) + tuple(cur[3: 5]) + tuple([comefrom])
                    else:
                        cur = tuple(cur[0:1]) + tuple([cur[1] - k * (args.code_length - 2)]) + tuple([cur[2] - k * (args.code_length - 2)]) + tuple(
                            cur[3: 5]) + tuple([comefrom])
                    cur_dfg.append(cur)
                else:
                    idx2 = idx1
                    break
            cur_dfg = cur_dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
            source_tokens += [x[0] for x in cur_dfg]
            position_idx += [0 for x in cur_dfg]
            source_ids += [tokenizer.unk_token_id for x in cur_dfg]
            padding_length = args.code_length + args.data_flow_length - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length

            dfg_to_dfg = [x[-1] for x in cur_dfg]
            dfg_to_code = [x[1:3] for x in cur_dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
            source_idss += source_ids
            position_idxs += position_idx
            # attention_mask
            attn_mask = np.zeros((args.code_length + args.data_flow_length,
                                  args.code_length + args.data_flow_length), dtype=bool)
            # calculate begin index of node and max length of input
            node_index = sum([i > 1 for i in position_idx])
            max_length = sum([i != 1 for i in position_idx])
            # sequence can attend to sequence
            attn_mask[:node_index, :node_index] = True
            # special tokens attend to all tokens
            for idx, i in enumerate(source_ids):
                if i in [0, 2]:
                    attn_mask[idx, :max_length] = True
            # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(dfg_to_code):
                if a < node_index and b < node_index:
                    attn_mask[idx + node_index, a:b] = True
                    attn_mask[a:b, idx + node_index] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(dfg_to_dfg):
                for a in nodes:
                    if a + node_index < len(position_idx):
                        attn_mask[idx + node_index, a + node_index] = True
            attn_mask = torch.tensor(attn_mask)
            if k == 0:
                attn_masks = attn_mask
            else:
                attn_masks = torch.cat((attn_masks, attn_mask), dim=0)
        i = 0
        subtrees = []
        for subtree in subtrees_tokens:
            if i + len(subtree) <= (args.code_length - 2) * args.code_snippets:
                i = i + len(subtree)
                subtrees.append(subtree)
            else:
                L = (args.code_length - 2) * args.code_snippets - i
                subtrees.append(subtree[:L])
                break

        features.append(
            InputFeatures(
                source_idss,
                position_idxs,
                attn_masks,
                example.label,
                example.idx,
                subtrees
            )
        )
    return features


def extract_data(tokenizer, args):
    train_examples = read_examples(args)
    train_features = convert_examples_to_features(train_examples, tokenizer, args)
    with open("dataset/PROMISE/" + args.project_name + ".pkl", 'wb') as f:
        pickle.dump(train_features, f)



def processing(project_name):
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type: e.g. roberta")
    ## Other parameters
    parser.add_argument("--train_filename", default="dataset/train/" + project_name, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--project_name", default=project_name, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--config_name", default="graphcodebert-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="graphcodebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--code_length", default=None, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=None, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_snippets", default=6, type=int,
                        help="The maximum length of BERT will limit the length of code that can be processed, and here "
                             "we recommend dividing long code into code snippets of a processable length.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, maxlength=512)

    extract_data(tokenizer, args)


if __name__ == "__main__":
    projects = ['ant-1.7', 'forrest-0.8', 'camel-1.6', 'ivy-2.0', 'jEdit-4.2', 'log4j-1.2', 'lucene-2.4', 'poi-3.0',
               'synapse-1.2', 'velocity-1.6.1', 'xalan-2.7', 'xerces-1.4.4']
    for project in projects:
        processing(project)
