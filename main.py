import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from logger import Logger

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Multi-Task ResNet-15')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--snapshot-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before taking a model snapshot')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--log-folder', type=str, default='./logs', metavar='LF',
                    help='where to store the training logs')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                    help='dataset for training(MNIST, fashionMNIST)')

def makeDataLoaders(args):
    train_dset = torchvision.datasets.FashionMNIST(root=args.data_folder,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
    test_dset = torchvision.datasets.FashionMNIST(root=args.data_folder,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dset,
                                            batch_size=args.test_batch_size,
                                            shuffle=True)
    return train_loader, test_loader

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)  
    
    def forward(self, x):
        x = x.view(-1, 784)
        out = self.fc1(x)
        return out

def train(model, criterion, optimizer, train_loader, logger, epoch, device, args, step):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        step += 1

        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        if (step+1) % args.log_interval == 0:
            print('Epoch [{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                .format(epoch + 1, batch_idx, len(train_loader), loss.item(), accuracy.item()))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = {'train loss': loss.item(), 'train accuracy': accuracy.item() }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), step)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

            # 3. Log training images (image summary)
            info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

            for tag, images in info.items():
                logger.image_summary(tag, images, step)
    return step

def test(model, criterion, test_loader, device, logger, step):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

            _, argmax = torch.max(outputs, 1)
            acc += (labels == argmax.squeeze()).float().mean()
    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # Log scalar values (scalar summary)
    info = {'test loss': test_loss, 'test accuracy': acc }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)

def snapshot(model, args, epoch):
    if (epoch + 1) % args.snapshot_interval == 0:
            if not os.path.exists(args.snapshot_folder):
                os.mkdir(args.snapshot_folder)
            torch.save(model.state_dict(), args.snapshot_folder + "/" + str(epoch+1) + '.pth')

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Device configuration
    device = torch.device("cuda" if args.cuda else "cpu")

    train_loader, test_loader = makeDataLoaders(args)

    model = NeuralNet().to(device)

    logger = Logger(args.log_folder)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        step = train(model, criterion, optimizer, train_loader, logger, epoch, device, args, step)
        test(model, criterion, test_loader, device, logger, step)
        snapshot(model, args, epoch)

if __name__ == "__main__":
    main()
