from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
batch_size = 2
iteration = 10000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/storage1/21721505/data/"
record_file = root_path
source1_name = "renji"
source2_name = 'luodian'
target_name = "huashan"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

def train(model):
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

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

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

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model):
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
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            #没有领域判别器，直接除以二
            pred = (pred1 + pred2) / 2
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
    return correct

if __name__ == '__main__':
    model = models.MFSAN(num_classes=2)
    print(model)
    if cuda:
        model.cuda()
    train(model)
