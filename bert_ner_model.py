import torch
import torch.nn as nn
from bert_base_model import BaseModel
from torchcrf import CRF
import config


class BertNerModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        out_dims = self.bert_config.hidden_size

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True,batch_first=True, dropout=args.dropout)
            self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.linear]
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
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1) #[batchsize, max_len, num_tags]
        else:
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
            # seq_out = self.dropout(seq_out)
            seq_out = self.classifier(seq_out)  # [24, 256, 53]

        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out, mask=attention_masks)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
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