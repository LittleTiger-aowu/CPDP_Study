import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, BCELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 1, config.hidden_size * 2)
        self.dense2 = nn.Linear(config.hidden_size * 2, config.hidden_size * 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.emb = nn.Embedding(44, 768)
        self.BatchNorm1d = nn.BatchNorm1d(config.hidden_size * 1)
        self.BatchNorm1d2 = nn.BatchNorm1d(config.hidden_size * 2)
        self.lstm1 = nn.LSTM(input_size=768, hidden_size=384, num_layers=1, bias=True, batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.tree = nn.Linear(768, 768)
        self.GRU = nn.GRU(input_size=768, hidden_size=768, num_layers=1, bias=True, dropout=0.1, bidirectional=False)
        self.GRU1 = nn.GRU(input_size=768, hidden_size=384, num_layers=1, bias=True, dropout=0.1, bidirectional=True)

    def forward(self, feature_s, source_idx, train_AST, feature_t=None, target_idx=None, test_AST=None):
        if feature_t is not None:
            for i in range(target_idx.shape[0]):
                idx = -1
                subtrees = test_AST[target_idx[i]]

                for k, subtree in enumerate(subtrees):
                    for l, path in enumerate(subtree):
                        idx +=1
                        vec = torch.tensor(path).to('cuda')
                        vec = self.emb(vec)
                        output, hn = self.GRU(vec)
                        vec = hn.flatten(0)
                        if l == 0:
                            leafs_vec = self.tree(feature_t[i, idx:idx + 1, :]) + vec
                        else:
                            leafs_vec = torch.cat((leafs_vec, self.tree(feature_t[i, idx:idx + 1, :]) + vec), dim=0)
                    output, hn = self.GRU1(leafs_vec)
                    output = torch.max(output, dim=0).values
                    sub_vec = torch.relu(output)
                    sub_vec = sub_vec.flatten(0).unsqueeze(0)
                    if k == 0 and l == 0:
                        subs_vec = sub_vec
                    else:
                        subs_vec = torch.cat((subs_vec, sub_vec), dim=0)
                temp, (hn, cn) = self.lstm1(subs_vec.unsqueeze(0))
                hn = hn.flatten(0).unsqueeze(0)
                cn = cn.flatten(0).unsqueeze(0)
                # split_code = torch.cat((hn, cn), dim=1)
                if i == 0:
                    Test_AST_hn = hn
                    Test_AST_cn = cn
                else:
                    Test_AST_hn = torch.cat((Test_AST_hn, hn), dim=0)
                    Test_AST_cn = torch.cat((Test_AST_cn, cn), dim=0)
            Test_AST = torch.cat((Test_AST_hn, Test_AST_cn), dim=1)

        for i in range(source_idx.shape[0]):
            idx = -1
            subtrees = train_AST[source_idx[i]]

            for k, subtree in enumerate(subtrees):
                for l, path in enumerate(subtree):
                    idx += 1
                    vec = torch.tensor(path).to('cuda')
                    vec = self.emb(vec)
                    output, hn = self.GRU(vec)
                    vec = hn.flatten(0)
                    if l == 0:
                        leafs_vec = self.tree(feature_s[i, idx:idx + 1, :]) + vec
                    else:
                        leafs_vec = torch.cat(
                            (leafs_vec, self.tree(feature_s[i, idx:idx + 1, :]) + vec), dim=0)
                output, hn = self.GRU1(leafs_vec)
                output = torch.max(output, dim=0).values
                sub_vec = torch.relu(output)
                sub_vec = sub_vec.flatten(0).unsqueeze(0)
                if k == 0 and l == 0:
                    subs_vec = sub_vec
                else:
                    subs_vec = torch.cat((subs_vec, sub_vec), dim=0)
                temp, (hn, cn) = self.lstm1(subs_vec.unsqueeze(0))
            hn = hn.flatten(0).unsqueeze(0)
            cn = cn.flatten(0).unsqueeze(0)
            OUTPUT = torch.max(temp, dim=1).values
            # split_code = torch.cat((hn, cn), dim=1)
            if i == 0:
                Train_AST_hn = hn
                Train_AST_cn = cn
                AST_OUTPUT = OUTPUT
            else:
                Train_AST_hn = torch.cat((Train_AST_hn, hn), dim=0)
                Train_AST_cn = torch.cat((Train_AST_cn, cn), dim=0)
                AST_OUTPUT = torch.cat((AST_OUTPUT, OUTPUT), dim=0)
        Train_AST = torch.cat((Train_AST_hn, Train_AST_cn), dim=1)

        x = AST_OUTPUT
        x = self.dropout(x)
        x = self.BatchNorm1d(x)
        x = self.dense(x)

        x = torch.relu(x)

        x = self.dropout(x)
        x = self.BatchNorm1d2(x)
        x = self.dense2(x)

        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        if feature_t is not None:
            return x, Train_AST, Test_AST, Train_AST_hn, Train_AST_cn, Test_AST_hn, Test_AST_cn
        else:
            return x


class Model(nn.Module):

    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, source_ids, source_position_idx, source_attn_mask, source_label, source_idx, train_AST,
                target_ids=None, target_position_idx=None, target_attn_mask=None, target_label=None, target_idx=None,
                test_AST=None):

        bs, l = source_ids.size()
        inputs_ids = source_ids.unsqueeze(1).view(bs * 1, l)
        position_idx = source_position_idx.unsqueeze(1).view(bs * 1, l)
        attn_mask = source_attn_mask

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        with torch.no_grad():
            for i in range(0, (self.args.code_length + self.args.data_flow_length) * self.args.code_snippets,
                           self.args.code_length + self.args.data_flow_length):
               inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(
                   inputs_ids[:, i:i + self.args.code_length + self.args.data_flow_length])  # [8,320,768]
               nodes_to_token_mask = nodes_mask[:, i:i + self.args.code_length + self.args.data_flow_length,
                                     None] & token_mask[:, None,
                                             i:i + self.args.code_length + self.args.data_flow_length] & attn_mask[:,
                                                                                                         i:i + self.args.code_length + self.args.data_flow_length,
                                                                                                         :]
               nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
               avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
               inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, i:i + self.args.code_length + self.args.data_flow_length,
                                                       None] + avg_embeddings * nodes_mask[:, i:i + self.args.code_length + self.args.data_flow_length, None]
               feature = self.encoder.roberta(inputs_embeds=inputs_embeddings,
                                              attention_mask=attn_mask[:, i:i + self.args.code_length + self.args.data_flow_length, :],
                                              position_ids=position_idx[:, i:i + self.args.code_length + self.args.data_flow_length],
                                              token_type_ids=position_idx[:, i:i + self.args.code_length + self.args.data_flow_length].eq(
                                                  -1).long()).last_hidden_state

               if i == 0:
                   feature_s = feature[:, 1: self.args.code_length - 1]
               else:
                   feature_s = torch.cat([feature_s, feature[:, 1: self.args.code_length - 1]], dim=1)


        if target_ids is not None:
            bs, l = target_ids.size()
            inputs_ids2 = target_ids.unsqueeze(1).view(bs * 1, l)  # [8,320]
            position_idx2 = target_position_idx.unsqueeze(1).view(bs * 1, l)  # [8,320]
            attn_mask2 = target_attn_mask
            nodes_mask2 = position_idx2.eq(0)  # [8,320]
            token_mask2 = position_idx2.ge(2)  # [8,320]
            with torch.no_grad():
                for i in range(0, (self.args.code_length + self.args.data_flow_length) * self.args.code_snippets,
                           self.args.code_length + self.args.data_flow_length):
                    inputs_embeddings2 = self.encoder.roberta.embeddings.word_embeddings(
                        inputs_ids2[:, i:i + self.args.code_length + self.args.data_flow_length])
                    nodes_to_token_mask2 = nodes_mask2[:, i:i + self.args.code_length + self.args.data_flow_length, None] & token_mask2[:, None,
                                                                                 i:i + self.args.code_length + self.args.data_flow_length] & attn_mask2[:,
                                                                                                  i:i + self.args.code_length + self.args.data_flow_length,
                                                                                                  :]
                    nodes_to_token_mask2 = nodes_to_token_mask2 / (nodes_to_token_mask2.sum(-1) + 1e-10)[:, :, None]
                    avg_embeddings2 = torch.einsum("abc,acd->abd", nodes_to_token_mask2, inputs_embeddings2)
                    inputs_embeddings2 = inputs_embeddings2 * (~nodes_mask2)[:, i:i + self.args.code_length + self.args.data_flow_length,
                                                              None] + avg_embeddings2 * nodes_mask2[:, i:i + self.args.code_length + self.args.data_flow_length,
                                                                                        None]
                    feature = self.encoder.roberta(inputs_embeds=inputs_embeddings2,
                                                   attention_mask=attn_mask2[:, i:i + self.args.code_length + self.args.data_flow_length, :],
                                                   position_ids=position_idx2[:, i:i + self.args.code_length + self.args.data_flow_length],
                                                   token_type_ids=position_idx2[:, i:i + self.args.code_length + self.args.data_flow_length].eq(
                                                       -1).long()).last_hidden_state

                    if i == 0:
                        feature_t = feature[:, 1: self.args.code_length - 1]
                    else:
                        feature_t = torch.cat([feature_t, feature[:, 1: self.args.code_length - 1]], dim=1)

            logits, Train_AST, Test_AST, Train_AST_hn, Train_AST_cn, Test_AST_hn, Test_AST_cn, = self.classifier(
                feature_s, source_idx, train_AST, feature_t, target_idx, test_AST)
            labels = source_label.unsqueeze(1)
            # shape: [batch_size, num_classes]
            prob = logits
            if labels is not None:
                loss_fct = BCELoss()
                loss = loss_fct(logits, labels.float())
                return loss, prob, Train_AST, Test_AST, Train_AST_hn, Train_AST_cn, Test_AST_hn, Test_AST_cn
            else:
                return prob
        logits = self.classifier(feature_s, source_idx, train_AST)
        labels = source_label.unsqueeze(1)
        # shape: [batch_size, num_classes]
        prob = logits
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(logits, labels.float())
            return loss, prob
        else:
            return prob
