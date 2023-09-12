import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from dataset import IEMCOAP_DATA 
from collections import defaultdict, OrderedDict
import argparse
import wandb

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
from models import Model, ALLformer, baseline
from tqdm import tqdm
import shutil
import datetime


parser = argparse.ArgumentParser(description='PSR-RL')
parser.add_argument("--text_dim", type=int, default=300)
parser.add_argument("--visual_dim", type=int, default=35)
parser.add_argument("--acoustic_dim", type=int, default=74)
parser.add_argument("--CT_fc_dim", type=int, default=128)
parser.add_argument("--CT_hidden_dim", type=int, default=128)
parser.add_argument("--CT_num_heads", type=int, default=8)
parser.add_argument("--CT_gru_dim", type=int, default=128)
parser.add_argument("--CT_out_dim", type=int, default=8)
parser.add_argument("--mode", type=str, default='negative')
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument('--lr', default=0.0001, type=float) #0.000075
parser.add_argument("--train_batch_size", type=int, default=256)
parser.add_argument("--dev_batch_size", type=int, default=256)
parser.add_argument("--test_batch_size", type=int, default=256)
parser.add_argument('--seed', default=1314, type=int)#8621 7059 645 9026 4301
parser.add_argument('--best', default=0.8332, type=float)
parser.add_argument('--path', default='/home/users/wsm/project/zhulei/dataset/iemocap_data.pkl', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--train_times', default=1, type=int)
parser.add_argument('--method', default='baseline', type=str)
args = parser.parse_args()

DEVICE = torch.device(f'cuda:{args.device}')


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

def main(args):
    #### make standard data from raw dataset and divide it into train, valid and test part,, just requires running in the first time
    set_random_seed(args.seed)

    current_time = datetime.datetime.now().time()
    current_date = datetime.date.today()

    save_path = os.path.join('path to save root',
                             'results',
                             'iemocap',
                             args.method,
                             'best_mean' + '_' + str(current_date) + '_' + str(current_time)
                             )
    os.makedirs(save_path, exist_ok=True)
    shutil.copy(
            'path to model.py',
            os.path.join(
                save_path,
                'model.py'
            )
        )
    
    train_dataset = IEMCOAP_DATA(args.path, 'train')
    dev_dataset = IEMCOAP_DATA(args.path, 'valid')
    test_dataset = IEMCOAP_DATA(args.path, 'test')
    
    train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=args.train_batch_size, 
                    shuffle=True, 
                    drop_last = False
                    )

    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    model = Model(args)
    # model = ALLformer()
    # model = baseline()

    model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', patience=20, factor=0.9, verbose=False)  # 50, 0.95
    model.to(DEVICE)


    best_ha, best_ha_f, best_sa, best_sa_f, best_an, best_an_f, best_ne, best_ne_f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch_i in tqdm(range(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, model_optimizer)
        valid_loss = eval_epoch(model, dev_dataloader)
        test_loss, accuracy, f1 = test_score_model(model, test_dataloader)
        model_scheduler.step(valid_loss)

        ne, ha, sa, an = accuracy
        ne_f, ha_f, sa_f, an_f = f1

        # print(
        #     '[Epoch %d] Training Loss: %.4f.    Valid Loss:%.4f.  Test Loss:%.4f.  Happy:%.4f.  Sad:%.4f.   Angry:%.4f.   Neutral:%.4f.    LR:%.4f.'
        #     % (epoch_i, train_loss, valid_loss, test_loss, ha_f, sa_f, an_f, ne_f,
        #        model_optimizer.param_groups[0]["lr"]))
        ##record best
        if best_ha <= ha:
            best_ha = ha
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('iemocap', "best_mae" + ".pth"))
        if best_ha_f <= ha_f:
            best_ha_f = ha_f
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))

        if best_sa <= sa:
            best_sa = sa
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
        if best_sa_f <= sa_f:
            best_sa_f = sa_f
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))

        if best_an <= an:
            best_an = an
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
        if best_an_f <= an_f:
            best_an_f = an_f
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))

        if best_ne <= ne:
            best_ne = ne
            save_data = {'model': model.state_dict()}
            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
        if best_ne_f <= ne_f:
            best_ne_f = ne_f
            save_data = {'model': model.state_dict()}

            # torch.save(save_data, os.path.join('cmu-mosi', "best_mae" + ".pth"))
    best_mean = (best_ha + best_sa + best_an + best_ne) / 4
    best_mean_f = (best_ha_f + best_sa_f + best_an_f + best_ne_f) / 4
    print('Best_Ha: %.4f.  Best_HaF:%.4f.  Best_Sa:%.4f.  Best_SaF:%.4f.   Best_An:%.4f.  Best_AnF:%.4f.     Best_Ne:%.4f.  Best_NeF:%.4f.'
          %(best_ha, best_ha_f, best_sa, best_sa_f, best_an, best_an_f, best_ne, best_ne_f))
    print('Best_Mean: %.4f.      Best_Mean_F: %.4f.' % (best_mean, best_mean_f))



    os.rename(
        save_path,
        save_path.replace('best_mean','Best_Mean: %.4f' % (best_mean))
    )

def train_epoch(model, train_dataloader, model_optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    tr_loss =  0

    for step, batch in enumerate(train_dataloader):
          
        batch = tuple(t.to(DEVICE) for t in batch)
        text, acoustic, visual, label_ids = batch
        label_ids = label_ids.squeeze()
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        b = visual.shape[0]

        out = model.forward(text, visual, acoustic)  # batch * 8
        pred = out.view(-1, 2)
        label = label_ids.view(-1)
        # loss = criterion(pred, label).mean()

        loss = criterion(pred[:b], label[:b]).mean() \
                + criterion(pred[b:2*b], label[b:b*2]).mean()\
                + criterion(pred[2*b:3*b], label[2*b:b*3]).mean() + \
                + criterion(pred[3*b:], label[3*b:]).mean()
    
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        tr_loss += loss
 
    return tr_loss/(step+1)

def eval_epoch(model,  dev_dataloader):
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):#, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            text, acoustic, visual, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            out = model(
                text,
                visual,
                acoustic
            )

            logits = out.view(-1, 2)
            label_ids = label_ids.view(-1)
            loss = nn.CrossEntropyLoss()(logits, label_ids)
            dev_loss += loss
    return dev_loss / (step + 1)

def test_epoch(model, test_dataloader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)

            text, acoustic, visual, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            label_ids = label_ids.squeeze()
            out = model.forward(text, visual, acoustic)

            preds.extend(out)
            labels.extend(label_ids)


        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        return preds, labels

def test_score_model(model, test_dataloader):

    preds, y_test = test_epoch(model, test_dataloader)
    test_loss = nn.CrossEntropyLoss()(preds.view(-1, 2), y_test.view(-1)).item()

    test_preds = preds.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = y_test.view(-1, 4).cpu().detach().numpy()
    f1, acc = [], []
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
        test_truth_i = test_truth[:,emo_ind]

        f1.append(f1_score(test_truth_i, test_preds_i, average='weighted'))
        acc.append(accuracy_score(test_truth_i, test_preds_i))
    
    return test_loss, acc, f1


if __name__ == '__main__':

    seed = args.seed
    for i in range(args.train_times):
        if seed == -1:
            args.seed = random.randint(0,99999)
        main(args)

