import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.metrics import f1_score
import random
from models import BertModel
from utils import get_tokenizer, MultimodalConfig
from dataset import MOSI_DATA
import datetime
import shutil

import warnings
warnings.filterwarnings('ignore')


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class cc_mosi(object):
    def __init__(self,args):
        self.args  = args
        set_random_seed(args.seed)


        self.save_path = args.save_path

        self.current_time = datetime.datetime.now().time()
        self.current_date = datetime.date.today()
        self.DEVICE = torch.device(f'cuda:{args.device}')

        self.save_path_ = os.path.join(
            self.save_path, args.dataset,
            args.method,
            'mae' + '_' + str(self.current_date) + '_' + str(self.current_time)
        )
        print(self.save_path_)
        os.makedirs(self.save_path_, exist_ok=True)
        
        shutil.copy(
            'path to model.py',
            os.path.join(
                self.save_path_,
                'model.py'
            )
        )
        self.init_dataloader()
        self.build_model()
        self.save_path = os.path.join(args.save_path, args.dataset)
        
 
    def train_epoch(self,epoch_i):
        self.model.train()
        criterion = nn.L1Loss()
        train_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            batch = tuple(t.to(self.DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            output = self.model(input_ids, visual, acoustic, \
                            token_type_ids=segment_ids, \
                            attention_mask=input_mask)
            loss = criterion(output, label_ids)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            train_loss += loss.item()

        return train_loss/(step+1)


    def valid(self,epoch_i):
        self.model.eval()
        valid_loss = 0
        criterion = nn.L1Loss()
        with torch.no_grad():
            for step, batch in enumerate(self.dev_dataloader):
                batch = tuple(t.to(self.DEVICE) for t in batch)
                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                output = self.model(input_ids, visual, acoustic, \
                            token_type_ids=segment_ids, \
                            attention_mask=input_mask)
                loss = criterion(output, label_ids)
                valid_loss += loss.item()
        return valid_loss/(step+1)

    def test(self,epoch_i, use_zero=False):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = tuple(t.to(self.DEVICE) for t in batch)
                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                output = self.model(input_ids, visual, acoustic, \
                            token_type_ids=segment_ids, \
                            attention_mask=input_mask)
                logits = output.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                logits = np.squeeze(logits).tolist()
                label_ids = np.squeeze(label_ids).tolist()
                preds.extend(logits)
                labels.extend(label_ids)
            preds = np.array(preds)
            labels = np.array(labels)
        
        preds = preds
        y_test = labels
        non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])
        
        preds = preds[non_zeros]
        y_test = y_test[non_zeros]

        mae = np.mean(np.absolute(preds - y_test))
        corr = np.corrcoef(preds, y_test)[0][1]

        preds = np.clip(preds, a_min=-3., a_max=3.)
        y_test = np.clip(y_test, a_min=-3., a_max=3.)
        mult = round(sum(np.round(preds) == np.round(y_test)) / float(len(y_test)), 5)  ### multi-class classification
        f_score_m = round(f1_score(np.round(preds), np.round(y_test), average='weighted'), 5)

        y_test_bin = (y_test >= 0)
        predictions_bin = (preds>= 0)  ###calculate binary acc
        bin = round(sum(y_test_bin == predictions_bin) / float(len(y_test)), 5)  ### multi-class classification
        f_score_b = round(f1_score(np.round(predictions_bin), np.round(y_test_bin), average='weighted'), 5)

        return mae,  corr, mult, f_score_m, bin, f_score_b

    def run(self):
        best_mae, best_corr, \
            best_mult, best_fscore_m, \
                best_bin, best_fscore_b = 100, 0, \
                                            0, 0, \
                                                0, 0
        for epoch_i in tqdm(range(int(self.args.n_epochs))):
            train_loss = self.train_epoch(epoch_i)
            valid_loss = self.valid(epoch_i)
            mae, corr, mult, f_score_m, bin, f_score_b = self.test(epoch_i)
            self.model_scheduler.step(valid_loss)
            # print(
            #     "epoch:{}, train_loss:{}, valid_loss:{}, test_accm:{}, test_accb:{}".format(
            #         epoch_i, train_loss, valid_loss, f_score_m, f_score_b
            #     )
            # )
                
            if best_mae >= mae:
                best_mae = mae
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
            if best_corr <= corr:
                best_corr = corr
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
            if best_mult <= mult:
                best_mult = mult
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
            if best_fscore_m <= f_score_m:
                best_fscore_m = f_score_m
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
            if best_bin <= bin:
                best_bin = bin
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
            if best_fscore_b <= f_score_b:
                best_fscore_b = f_score_b
                # save_data = {'model': model.state_dict(), 'actor': actor.state_dict()}
                # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))


        print(
                'Best_MAE: %.4f.  Best_Corr:%.4f.  Best_Mult:%.4f.  Best_Fscore:%.4f.   Best_Bin:%.4f.  Best_Fscore_B:%.4f.' % (
                best_mae, best_corr, best_mult, best_fscore_m, best_bin, best_fscore_b))

        with open(os.path.join(self.save_path_,'log.txt'),'a') as f:
            info = '\n'
            info += f'\nseed is {args.seed}'
            info += 'Best_MAE: %.4f.  Best_Corr:%.4f.  Best_Mult:%.4f.  Best_Fscore:%.4f.   Best_Bin:%.4f.  Best_Fscore_B:%.4f.' % (
                best_mae, best_corr, best_mult, best_fscore_m, best_bin, best_fscore_b)
            f.write(info)
        os.rename(
            self.save_path_,
            self.save_path_.replace('mae','%.4f'%(best_mae))
        )




    def init_wandb(self):
        wandb.init(project="cc", entity="cc")
        wandb.config.update(self.args)
    
    def init_dataloader(self):
        if self.args.dataset == 'mosei':
            self.args.VISUAL_DIM = 35
            self.args.ACOUS_DIM = 74
            self.args.dim_v = 35
            self.args.dim_a = 74
            # self.args.
        mosi_data = MOSI_DATA(self.args)
        self.train_dataloader, \
            self.dev_dataloader, \
                self.test_dataloader = mosi_data.create_dataloader()

    def build_model(self):
        multimodal_config = MultimodalConfig(
                    dropout_prob=self.args.dropout_prob, 
                    emb_size=self.args.emb_size, 
                    dim_t=self.args.dim_t,
                    dim_a=self.args.dim_a, 
                    dim_v=self.args.dim_v, 
                    seqlength= self.args.max_seq_length,
                    mode = self.args.mode
                    )

        self.model = BertModel.from_pretrained(
                        self.args.model, 
                        multimodal_config=multimodal_config, 
                        num_labels=1,
                    )

        self.model.to(self.DEVICE)

        # Prepare optimizer
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # self.model_scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=20, gamma=0.9) # 20, 0.95
        self.model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, mode='min', patience=20, factor=0.9, verbose=False) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--dim_t", type=int, default=768)  #768
    parser.add_argument("--dim_a", type=int, default=74)   #74
    parser.add_argument("--dim_v", type=int, default=47)   #47
    parser.add_argument("--train_batch_size", type=int, default=250)
    parser.add_argument("--dev_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--dropout_prob", type=float, default=0.15)
    parser.add_argument("--mode", type=str, default='positive')

    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased"],
        default="bert-base-uncased",
    )
    parser.add_argument("--ACOUS_DIM", default=74, type=str)
    parser.add_argument("--VISUAL_DIM", default=47, type=str)
    parser.add_argument("--TEXT_DIM", default=768, type=str)
    parser.add_argument("--device",default=0, type=int)
    parser.add_argument("--learning_rate", type=float, default=0.00005)####0.00008
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--train_times", default=1, type=int)
    parser.add_argument("--save_path", default="/home/users/wsm/project/zhulei/cc/results",type=str)
    parser.add_argument("--dataset", type=str,
                        choices=["mosi", "mosei"], default="mosei")
    parser.add_argument('--method', default='baseline', type=str)

    args = parser.parse_args()
    seed = args.seed
    for i in range(args.train_times):
        if seed == -1:
            args.seed = random.randint(0, 10000)
        cc = cc_mosi(args)
        cc.run()
