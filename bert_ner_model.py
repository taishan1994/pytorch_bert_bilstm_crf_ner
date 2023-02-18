import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from bert_base_model import BaseModel
from transformers import AutoConfig, AutoModel, BertModel
from torchcrf import CRF
import config



class LayerNorm(nn.Module):
    def __init__(self, filters, elementwise_affine=False):
        super(LayerNorm, self).__init__()
        self.LN = nn.LayerNorm([filters],elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.LN(x)
        return out.permute(0, 2, 1)



class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}
        ]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(filters) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(filters) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class NormalNerModel(nn.Module):
    def __init__(self,
                 args,
                 **kwargs):
        super(NormalNerModel, self).__init__()
        config = AutoConfig.from_pretrained(args.bert_dir)
        vocab_size = config.vocab_size
        out_dims = config.hidden_size
        self.embedding = nn.Embedding(vocab_size, out_dims)

        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        init_blocks = []
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
        init_blocks.append(self.linear)

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True, batch_first=True,
                                dropout=args.dropout)
        elif args.use_idcnn == "True":
            self.idcnn = IDCNN(out_dims, args.lstm_hidden * 2)

        if args.use_crf == 'True':
            if args.model_name.split('_')[0] == "crf":
                self.crf_linear = nn.Linear(out_dims, args.num_tags)
                init_blocks = [self.crf_linear]
            self.crf = CRF(args.num_tags, batch_first=True)

        self._init_weights(init_blocks, initializer_range=0.02)

    def _init_weights(self, blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels,
                word_ids=None):

        seq_out = self.embedding(token_ids)

        batch_size = seq_out.size(0)

        if self.args.use_idcnn == "True":
            seq_out = self.idcnn(seq_out)
            seq_out = self.linear(seq_out)
        elif self.args.use_lstm == 'True':
            # hidden = self.init_hidden(batch_size)
            # seq_out, (hn, _) = self.lstm(seq_out, hidden)
            # seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out, _ = self.lstm(seq_out)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1)  # [batchsize, max_len, num_tags]

        if self.args.use_crf == 'True':
            if self.args.model_name.split('_')[0] == "crf":
                seq_out = self.crf_linear(seq_out)
            logits = self.crf.decode(seq_out, mask=attention_masks)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss,) + (logits,)
            return outputs
        else:
            logits = seq_out
            if labels is None:
                return logits
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss,) + (logits,)
            return outputs


class BertNerModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob, model_name=args.model_name)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        out_dims = self.bert_config.hidden_size

        init_blocks = []
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
        init_blocks.append(self.linear)

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True,batch_first=True, dropout=args.dropout)
        elif args.use_idcnn == "True":
            self.idcnn = IDCNN(out_dims, args.lstm_hidden * 2)
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            #
            out_dims = mid_linear_dims

            # self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(out_dims, args.num_tags)
            # self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.criterion = nn.CrossEntropyLoss()


            init_blocks = [self.mid_linear, self.classifier]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

        if args.use_crf == 'True':
            self.crf = CRF(args.num_tags, batch_first=True)

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)

        if self.args.use_lstm == 'True':
            # hidden = self.init_hidden(batch_size)
            # seq_out, (hn, _) = self.lstm(seq_out, hidden)
            # seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out, _ = self.lstm(seq_out)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1) #[batchsize, max_len, num_tags]
        elif self.args.use_idcnn == "True":
            seq_out = self.idcnn(seq_out)
            seq_out = self.linear(seq_out)
        else:
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
            # seq_out = self.dropout(seq_out)
            seq_out = self.classifier(seq_out)  # [24, 256, 53]

        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out, mask=attention_masks)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss, ) + (logits,)
            return outputs
        else:
            logits = seq_out
            if labels is None:
                return logits
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss,) + (logits,)
            return outputs

if __name__ == '__main__':
    args = config.Args().get_parser()
    args.num_tags = 33
    args.use_lstm = 'True'
    args.use_crf = 'True'
    model = BertNerModel(args)
    for name,weight in model.named_parameters():
        print(name)
