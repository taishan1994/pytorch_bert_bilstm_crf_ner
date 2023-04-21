import sys
sys.path.append('..')
import torch
import torch.nn as nn
# from torchcrf import CRF
from layers.CRF import CRF
from transformers import BertModel

class BertNerModel(nn.Module):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__()
        self.bert_module = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True,
                              hidden_dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        self.bert_config = self.bert_module.config
        out_dims = self.bert_config.hidden_size
        init_blocks = []
        self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
        init_blocks.append(self.linear)

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True,batch_first=True, dropout=args.dropout)
            
            self.criterion = nn.CrossEntropyLoss()
            # init_blocks = [self.linear]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
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
                input_ids,
                attention_mask,
                token_type_ids,
                ):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        if self.args.use_lstm == 'False':
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
            # seq_out = self.dropout(seq_out)
            seq_out = self.classifier(seq_out)  # [24, 256, 53]

        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out)
            return torch.tensor(logits)
        if self.args.use_crf == 'False':
            logits = seq_out
            return logits

