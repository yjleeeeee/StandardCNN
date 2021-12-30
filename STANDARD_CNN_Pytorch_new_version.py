import argparse
import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import util
import torch._utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST]')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=15, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-load', default='', type=str, metavar='PATH',
                    help='path to training mask (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
best_prec1 = 0
change = 64
change2 = 128
change3 = 164
change4 = 224

tp1 = [];
tp5 = [];
ep = [];
lRate = [];
device_num = 1
scale = 1.5

tp1_tr = [];
tp5_tr = [];
losses_tr = [];
losses_eval = [];

def main():
    global args, best_prec1, batch_size, device_num, scale

    args = parser.parse_args()
    batch_size = args.batch_size
    numCls = 10
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(42)

    device_num = 1
    if device_num < 2:
        device = 0 # 0
        torch.cuda.set_device(device)
       # model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    model = ConvNet(numCls)

    model.cuda()
    
    #criterion = torch.nn.MSELoss(size_average=False).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_en = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0) # Adam-->SGD weight_decay change

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = torchvision.datasets.CIFAR10('./data_CIFAR10', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_data = torchvision.datasets.CIFAR10('./data_CIFAR10', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_data,  # val_data for testing
                                             batch_size=int(args.batch_size), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    if args.evaluate:
        prec1 = validate(val_loader, model, criterion, criterion_en) # time step remove
       # return

   # prec1_tr = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        ep.append(epoch)

        train(train_loader, model, criterion, criterion_en, optimizer, epoch) # time_step=100 remove
        prec1 = validate(val_loader, model, criterion, criterion_en)# time_step=100 remove

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} \t'
              'En_Loss_Eval {losses_en_eval: .4f} \t'
              'Prec@1_tr {top1_tr:.3f} \t'
              'Prec@5_tr {top5_tr:.3f} \t'
              'En_Loss_train {losses_en: .4f}'.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k], losses_en_eval=losses_eval[k], top1_tr=tp1_tr[k],
            top5_tr=tp5_tr[k], losses_en=losses_tr[k]))



def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred),1))

def train(train_loader, model, criterion, criterion_en, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_tr = AverageMeter()
    top5_tr = AverageMeter()
    losses_en = AverageMeter()

    # switch to train mode
    model.train()

    # define the binarization operator
   # bin_op = util.BinOp(model)

    end = time.time()
    # print ('mark1',train_loader.sampler)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        labels = Variable(target.cuda())
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input)

        # Binarization
    #    bin_op.binarization()

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters
        output = model(input_var)

        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss = criterion(output, labels)
        loss_en = criterion_en(output, labels).cuda()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        losses_en.update(loss_en.item(), input.size(0))
        top1_tr.update(prec1_tr.item(),  input.size(0))
        top5_tr.update(prec5_tr.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        # # restore weights
     #  bin_op.restore()
     #   bin_op.updateBinaryGradWeight()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)


def validate(val_loader, model, criterion, criterion_en, time_step=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_best = AverageMeter()
    top5 = AverageMeter()
    losses_en_eval = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # define the binarization operator
   # bin_op = util.BinOp(model)

    # Binarization
   # bin_op.binarization()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = Variable(target.cuda())
        target = target.cuda()
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input)
        if time_step == 0:
            output = model(input_var)
        else:
            output = model(input_var, steps=time_step)
       
       # output = model.tst(input=input_var, steps = time_step)

        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss = criterion(output, labels)
        loss_en = criterion_en(output, labels).cuda()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(),  input.size(0))
        losses_en_eval.update(loss_en.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

   # bin_op.restore()

    print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1, top5=top5, losses_en_eval=losses_en_eval))

    tp1.append(top1.avg)
    tp5.append(top5.avg)
    losses_eval.append(losses_en_eval.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpointT1_vgg8_xnor__2021.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_bestT1_vgg8_xnor_2021.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr  

    for param_group in optimizer.param_groups:
        if epoch >= change4:
            param_group['lr'] = 0.2*0.2*0.2*0.2*lr

        if epoch >= change3:
            param_group['lr'] = 0.2 * 0.2 * 0.2 * lr

        elif epoch >= change2:
            param_group['lr'] = 0.2 * 0.2 * lr

        elif epoch >= change:
            param_group['lr'] = 0.2 * lr

        else:
            param_group['lr'] = lr

    lRate.append(param_group['lr'])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#class BinaryActive(torch.autograd.Function):
#    def forward(self, input):
#        self.save_for_backward(input)
#        size = input.size()
#        input = input.sign()
#         return input

#    def backward(self, grad_output):
#        input, = self.saved_tensors
#        grad_input = grad_output.clone()
#        grad_input[input.ge(1)] = 0
#        grad_input[input.le(-1)] = 0
#        return grad_input


#class BinaryConv2d(nn.Module):
 #   def __init__(self, input_channels, output_channels,
 #           kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
 #           Linear=False):
 #       super(BinaryConv2d, self).__init__()
 #       self.layer_type = 'BinaryConv2d'
 #       self.kernel_size = kernel_size
 #       self.stride = stride
 #       self.padding = padding
 #       self.dropout_ratio = dropout

  #      if dropout!=0:
  #          self.dropout = nn.Dropout(dropout)
   #          self.Linear = Linear
    #    if not self.Linear:
     #       self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
      #      self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
       # else:
       #     self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
       #     self.linear = nn.Linear(input_channels, output_channels)
       # self.relu = nn.ReLU(inplace=True)

  #  def forward(self, x):
  #      x = self.bn(x)
  #      x = BinaryActive.apply(x)
  #      if self.dropout_ratio!=0:
  #          x = self.dropout(x)
  #      if not self.Linear:
  #          x = self.conv(x)
  #      else:
  #          x = self.linear(x)
  #      x = self.relu(x)
  #      return x


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.xnor = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
           # nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.13),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.13),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.13),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(p=0.13),
           # nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=True),
            nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n1 = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2. / (n1))
                m.weight.data.normal_(0, variance1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / (fan_in))
                m.weight.data.normal_(0, variance2)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 512 * 4 * 4)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    main()
