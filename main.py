import os
import logging
import numpy as np
import torch
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import config
import dataset
# 要显示传入BertFeature
from preprocess import BertFeature
import bert_ner_model
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)



class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        model = bert_ner_model.BertNerModel(args)
        self.model, self.device = trainUtils.load_model_and_parallel(model, args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs
        self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 90 #每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)
                loss, logits = self.model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'], batch_data['labels'])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1
                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                    if f1_score > best_f1:
                        trainUtils.save_model(self.args, self.model, model_name + '_' + args.data_name, global_step)
                        best_f1 = f1_score
    def dev(self):
        self.model.eval()
        with torch.no_grad():
            batch_output_all = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_loss, dev_logits = self.model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'], dev_batch_data['labels'])
                tot_dev_loss += dev_loss.item()
                if self.args.use_crf == 'True':
                    batch_output = dev_logits
                else:
                    batch_output = dev_logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)
                if len(batch_output_all) == 0:
                    batch_output_all = batch_output
                else:
                    batch_output_all = np.append(batch_output_all, batch_output, axis=0)
            total_count = [0 for _ in range(len(label2id))]
            role_metric = np.zeros([len(id2label), 3])
            for pred_label, tmp_callback in zip(batch_output_all, dev_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(id2label), 3])
                pred_entities = decodeUtils.bioes_decode(pred_label[1:1 + len(text)], text, self.idx2tag)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            # print('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def test(self, model_path):
        model = bert_ner_model.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_label = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'],dev_batch_data['labels'])
                if self.args.use_crf == 'True':
                    batch_output = logits
                else:
                    batch_output = logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)
                if len(pred_label) == 0:
                    pred_label = batch_output
                else:
                    pred_label = np.append(pred_label, batch_output, axis=0)
            total_count = [0 for _ in range(len(id2label))]
            role_metric = np.zeros([len(id2label), 3])
            for pred, tmp_callback in zip(pred_label, dev_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(id2label), 3])
                pred_entities = decodeUtils.bioes_decode(pred[1:1 + len(text)], text, self.idx2tag)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric
            logger.info(metricsUtils.classification_report(role_metric, label_list, id2label, total_count))

    def predict(self, raw_text, model_path):
        model = bert_ner_model.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
            encode_dict = tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding='max_length',
                                    truncation='longest_first',
                                    is_pretokenized=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
            if self.args.use_crf == 'True':
                output = logits
            else:
                output = logits.detach().cpu().numpy()
                output = np.argmax(output, axis=2)
            pred_entities = decodeUtils.bioes_decode(output[0][1:1 + len(tokens)], "".join(tokens), self.idx2tag)
            logger.info(pred_entities)


if __name__ == '__main__':
    data_name = args.data_name
    #data_name = 'attr'
    #args.train_epochs = 3
    #args.train_batch_size = 32
    #args.max_seq_len = 150
    model_name = ''
    if args.use_lstm == 'True' and args.use_crf == 'False':
        model_name = 'bert_bilstm'
    if args.use_lstm == 'True' and args.use_crf == 'True':
        model_name = 'bert_bilstm_crf'
    if args.use_lstm == 'False' and args.use_crf == 'True':
        model_name = 'bert_crf'
    if args.use_lstm == 'False' and args.use_crf == 'False':
        model_name = 'bert'

    commonUtils.set_logger(os.path.join(args.log_dir, '{}_{}.log'.format(model_name, args.data_name)))
    
    if data_name == "cner":
        args.data_dir = './data/cner'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        test_dataset = dataset.NerDataset(test_features)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        
        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, test_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
        bertForNer.test(model_path)
        
        raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)

    if data_name == "chip":
        args.data_dir = './data/CHIP2020'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)
        
        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
        bertForNer.test(model_path)
        
        raw_text = "大动脉转换手术要求左心室流出道大小及肺动脉瓣的功能正常，但动力性左心室流出道梗阻并非大动脉转换术的禁忌证。"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)

    if data_name == "clue":
        args.data_dir = './data/CLUE'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)
        
        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
        bertForNer.test(model_path)
        
        raw_text = "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)
    if data_name == "addr":
        args.data_dir = './data/addr'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)
        
        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        # bertForNer.train()

        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
        # bertForNer.test(model_path)
        
        raw_text = "浙江省嘉兴市平湖市钟埭街道新兴六路法帝亚洁具厂区内万杰洁具"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)

    if data_name == "attr":
        args.data_dir = './data/attr'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)
        
        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
        bertForNer.test(model_path)
        
        raw_text = "荣耀V9Play支架手机壳honorv9paly手机套新品情女款硅胶防摔壳"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)
    
