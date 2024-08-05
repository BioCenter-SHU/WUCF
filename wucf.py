from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source1', type=str)
    parser.add_argument('--source2', type=str) 
    parser.add_argument('--target', type=str)
    parser.add_argument('--rootpath', type=str,default='./')
    
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    

    return args

# Training settings
batch_size = 8
iteration = 4000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = args.root_path
record_file = root_path
source1_name = args.source1
source2_name = args.source2
target_name = args.target

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

def train(model,record_file):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': lr[1]},
            {'params': model.cls_fc_son2.parameters(), 'lr': lr[1]},
            {'params': model.sonnet1.parameters(), 'lr': lr[1]},
            {'params': model.sonnet2.parameters(), 'lr': lr[1]},
            {'params': model.domain_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)


    for i in range(1, iteration + 1):
        model.train()

        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, dm_loss = model(source_data, target_data, source_label, mark=0, batch_size = batch_size)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * (mmd_loss + dm_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tdm_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), dm_loss.item()))

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, dm_loss = model(source_data, target_data, source_label, mark=1, batch_size = batch_size)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + dm_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tdm_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), dm_loss.item()))
        
        if record_file:
                record = open(record_file+source1_name+'_'+source2_name+'_'+target_name+'_'+'loss.txt', 'a')
                record.write('%s %s %s\n' % (cls_loss.data, mmd_loss.data, dm_loss.data))
                record.close()

        if i % (log_interval * 20) == 0:
            t_correct = test(model,record_file=record_file)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model,record_file):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metric_collection = MetricCollection({
                'acc': Accuracy(task='multiclass', num_classes=2, average="micro").to(device),
                'prec': Precision(task='multiclass',num_classes=2, average='macro').to(device),
                'rec': Recall(task='multiclass',num_classes=2, average='macro').to(device)
            })
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred_dm = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred_dm*pred1 + pred_dm*pred2)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            metric_collection.forward(pred,target)
            val_metrics = metric_collection.compute()
            #classifier 1
        acc = val_metrics['acc']
        prec = val_metrics['prec']
        rec = val_metrics['rec']

        test_loss /= len(target_test_loader.dataset)
        print('\nTest set: Average loss: {:.4f},  Accuracy: ({:.4f}%) Precision: ({:.4f}%) Recall:  ({:.4f}%)\n'.format(
                    test_loss, 100*acc, 100*prec, 100*rec))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
        if record_file:
                    record = open(record_file+source1_name+'_'+source2_name+'_'+target_name+'_'+'acc.txt', 'a')
                    record.write('%s %s %s\n' % (acc.data, prec.data, rec.data))
                    record.close()
    return correct


args = get_args()
model = models.MADA(num_classes=2, domain_classes=2)
print(model)
if cuda:
    model.cuda()

train(model, record_file = record_file)

import os
# 设置文件名
Accuracy_file = record_file+"acc.txt"
Loss_file = record_file+"loss.txt"
if  not os.path.exists(Accuracy_file):
    open(Accuracy_file, "w")
if  not os.path.exists(Loss_file):
    open(Loss_file, "w")

