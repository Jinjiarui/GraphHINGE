import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import gc
sys.path.append("..")
from utils import data_loader,utils
import argparse
import random
import pickle as pkl
import tqdm
import datetime
from time import time
from algo import GraphHINGE_FFT,GraphHINGE_Conv,GraphHINGE_Cross,GraphHINGE_CrossConv,GraphHINGE_ALL
#from tensorboardX import SummaryWriter
import numpy as np

def train(model, device, optimizer, train_loader, eval_loader, save_dir, epochs, log_file, model_name, patience):
    train_dir = save_dir+'/Logs/'+model_name+'_train'
    eval_dir = save_dir+'/Logs/'+model_name+'_eval'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    #train_writer = SummaryWriter(log_dir=train_dir)
    #eval_writer = SummaryWriter(log_dir=eval_dir)
    best_val_acc = 0.0
    cnt = 0
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("Epoch {} >>".format(epoch+1))
        t0=time()
        #train
        model.train()
        train_loss = []
        for i, data in enumerate(train_loader):
            UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
            optimizer.zero_grad()
            pred = model(UI.to(device), IU.to(device),\
            UIUI.to(device), IUIU.to(device), \
            UIAI1.to(device), IAIU1.to(device), \
            UIAI2.to(device), IAIU2.to(device), \
            UIAI3.to(device), IAIU3.to(device))

            loss = criterion(pred, labels.to(device))
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            '''
            train_iter= epoch * len(train_loader) + i
            if train_iter%10 ==0:
                train_writer.add_scalar('train_loss', loss.item(), train_iter)
                
            if i%500==0:
                print('Epoch {:d} | Batch {:d} | Train Loss {:.4f} | '.format(epoch + 1, i+1,loss))
                log_file.write('Epoch {:d} | Batch {:d} | Train Loss {:.4f} | '.format(epoch + 1, i+1,loss))
                

            del UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            '''

        train_loss = np.mean(train_loss) 
        print('Epoch {:d} | Train Loss {:.4f} | '.format(epoch + 1, train_loss))
        #log_file.write('Epoch {:d} | Train Loss {:.4f} | '.format(epoch + 1, train_loss))
        
        t1=time()
        print("Train time: {}".format(t1-t0))
        t0=time()
        #eval
        model.eval()
        eval_loss = []
        eval_acc = []
        eval_auc = []
        eval_logloss = []
        eval_f1 = []
        with torch.no_grad():
            t0=time()
            for i, data in enumerate(eval_loader):
                UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
                
                pred = model(UI.to(device), IU.to(device),\
                UIUI.to(device), IUIU.to(device), \
                UIAI1.to(device), IAIU1.to(device), \
                UIAI2.to(device), IAIU2.to(device), \
                UIAI3.to(device), IAIU3.to(device))

                loss = criterion(pred, labels.to(device))
                acc = utils.evaluate_acc(pred.detach().cpu().numpy(), labels.numpy())
                eval_loss.append(loss.item())
                eval_acc.append(acc)

                '''
                del UI,IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                '''

            eval_loss = np.mean(eval_loss)
            eval_acc = np.mean(eval_acc)

            t1=time()
            print("Eval time: {}".format(t1-t0))

            print('Epoch {:d} | Eval Loss {:.4f} | Eval ACC {:.4f} | '.format(epoch + 1, eval_loss, eval_acc))
            #log_file.write('Epoch {:d} | Eval Loss {:.4f} | Eval ACC {:.4f} | '.format(epoch + 1, eval_loss, eval_acc))
            #eval_writer.add_scalar('eval_loss', eval_loss, train_iter)
            #eval_writer.add_scalar('eval_acc', eval_acc, train_iter)
            if eval_acc > best_val_acc:
                best_val_acc = eval_acc
                best_epoch = epoch
                #log_file.write("Saving best weights...\n")
                torch.save(model.state_dict(), os.path.join(save_dir,model_name))
                cnt = 0
            else: 
                cnt = cnt + 1
                if cnt >= patience:
                    print("Early stopping")
                    print("best epoch:{}".format(best_epoch))
                    break
            

def test(model, device, test_loader):
    test_dir = save_dir+'/Logs/'+model_name+'_test'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    #test_writer = SummaryWriter(log_dir=test_dir)
    test_loss = []
    test_auc = []
    test_acc = []
    test_logloss = []
    test_f1 = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
            pred = model(UI.to(device), IU.to(device),\
            UIUI.to(device), IUIU.to(device), \
            UIAI1.to(device), IAIU1.to(device), \
            UIAI2.to(device), IAIU2.to(device), \
            UIAI3.to(device), IAIU3.to(device))
        
            loss = criterion(pred, labels.to(device))

            auc = utils.evaluate_auc(pred.detach().cpu().numpy(), labels.numpy())
            acc = utils.evaluate_acc(pred.detach().cpu().numpy(), labels.numpy())
            f1 = utils.evaluate_f1_score(pred.detach().cpu().numpy(), labels.numpy())
            logloss = utils.evaluate_logloss(pred.detach().cpu().numpy(), labels.numpy())
            test_loss.append(loss.item())
            test_auc.append(auc)
            test_acc.append(acc)
            test_f1.append(f1)
            test_logloss.append(logloss)
            '''
            test_writer.add_scalar('test_loss', loss.item(), i)
            test_writer.add_scalar('test_auc', auc, i)
            test_writer.add_scalar('test_acc', acc, i)
            test_writer.add_scalar('test_f1', f1, i)
            test_writer.add_scalar('test_logloss', logloss, i)
            
            del UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
            gc.collect()
            
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            '''
            
        
        test_loss = np.mean(test_loss)
        test_auc = np.mean(test_auc)
        test_acc = np.mean(test_acc)
        test_f1 = np.mean(test_f1)
        test_logloss = np.mean(test_logloss)
        print('Test Loss {:.4f} | Test AUC {:.4f} | Test ACC {:.4f} | Test F1 {:.4f} | Test Logloss {:.4f} |'.format(test_loss, test_auc, test_acc, test_f1, test_logloss))
        #log_file.write('Test Loss {:.4f} | Test AUC {:.4f} | Test ACC {:.4f} | Test F1 {:.4f} | Test Logloss {:.4f} |'.format(test_loss, test_auc, test_acc, test_f1, test_logloss))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('-cuda', type=int, default=0 ,help='Cuda number.')
    parser.add_argument('-hidden', type=int, default=128, help='Hidden Units.')
    parser.add_argument('-heads', type=int, default=3, help='Attention heads.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('-wd', type=float, default=0.0009, help='Weight decay.')
    parser.add_argument('-epochs', type=int, default=500,help='Maximum Epoch.')

    parser.add_argument('-model', type=str, default='GraphHINGE_FFT',help='Model.')
    parser.add_argument('-save_dir', type=str, default='../out',help='Trained models to be saved.')
    parser.add_argument('-pretrained', type=str, default='pretrained.pth',help='Pretrained model.')
    parser.add_argument('-model_num', type=str, default='0',help='Model num.')

    parser.add_argument('-d', type=str,help='Dataset.')
    parser.add_argument('-p', type=str,help='Dataset path.')
    parser.add_argument('-o', type=str,default='../data/data_out',help='Output path.')
    parser.add_argument('-b', type=int, default=128, help='Batch size.')
    parser.add_argument('-s', type=int,default=0, help='Sample path to be saved.')
    parser.add_argument('-n', type=int, default=16,help='Num of walks per node.')
    parser.add_argument('-w', type=int, default=1,help='Scale of walk length.')
    parser.add_argument('-temp1', type=float, default=0.2, help='Temperature factor for node att.')
    parser.add_argument('-temp2', type=float, default=0.2, help='Temperature factor for path att.')
    parser.add_argument('-ratio', type=float, default=1, help='Sample ratio.')

    args = parser.parse_args() 
    device = torch.device("cuda:" + str(args.cuda))
    user_metas, item_metas, train_loader, eval_loader, test_loader, user_num, item_num, attr1_num, attr2_num, attr3_num = \
    data_loader.Dataloader(args.p, args.d, args.o, args.s, args.n, args.w, args.b,ratio=args.ratio).load_data()
    print('Data Loaded!')

    if args.model == 'GraphHINGE_FFT':
        model = GraphHINGE_FFT.GraphHINGE(
            user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
            ).to(device)
    elif args.model == 'GraphHINGE_Conv':
        model = GraphHINGE_Conv.GraphHINGE(
            user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
            ).to(device)
    elif args.model == 'GraphHINGE_Cross':
        model = GraphHINGE_Cross.GraphHINGE(
            user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
            ).to(device)
    elif args.model == 'GraphHINGE_CrossConv':
        model = GraphHINGE_CrossConv.GraphHINGE(
            user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
            ).to(device)
    elif args.model == 'GraphHINGE_ALL':
        model = GraphHINGE_ALL.GraphHINGE(
            user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
            ).to(device)
    else:
        raise NotImplementedError
    print('Model Created!')
    
    if (os.path.exists(os.path.join(args.save_dir, args.pretrained))):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.pretrained)))
        print("Model loaded!")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    log_name = args.model + '_'+ args.d + '_'+ args.model_num +'.txt'
    log_file = open(os.path.join(args.save_dir, log_name), "w+")
    model_name = args.model + '_'+ args.d + '_'+ args.model_num +'.pth'
    save_dir = args.save_dir
    train(model, device, optimizer, train_loader, eval_loader, args.save_dir, args.epochs, log_file, model_name,patience=25)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, model_name)))
    test(model, device, test_loader)