python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=33 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=3 \
--eval_batch_size=32 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='True' \
--use_crf='False' \
--dropout_prob=0.3 \
--dropout=0.3 \