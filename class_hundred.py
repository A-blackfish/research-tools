# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

sourced from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Modified by : Harold Mouchère / University of Nantes

2018

"""
import time
import copy
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import torch.nn as nn
import torch.nn.functional as F


def main():
    path = 'E:\\document\\Fan_dc\\TP-MLP-master-629e1792db78518cbb8c62ecfaf0db5db9724ba9\\data\\CROHME'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    minibatchsize = 32
    test_size = 100
    save_model = './Hundred_class_2.nn'
    num_epochs = 20

    # Assume that we are on a CUDA machine, then this should print a CUDA device:

    print(device)

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose(
        [transforms.Grayscale(),  # CROHME png are RGB, but already 32x32
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    fulltrainset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    # split the full train part as train, validation and test, or use the 3 splits defined in the competition
    a_part = int(len(fulltrainset) / 5)
    trainset, validationset, testset = torch.utils.data.random_split(fulltrainset, [3 * a_part, a_part,
                                                                                    len(fulltrainset) - 4 * a_part])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatchsize,
                                              shuffle=True, drop_last=True, num_workers=1)

    validationloader = torch.utils.data.DataLoader(validationset, batch_size=minibatchsize,
                                                   shuffle=False, drop_last=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_size,
                                             shuffle=False, drop_last=True, num_workers=0)



    # define the set of class names :
    classes = [x[0].replace(path, '') for x in os.walk(path)][
              1:]  # all subdirectories, except itself
    print('Name of classes:',classes)
    nb_classes = len(classes)
    print("Number of classes :%d , training size:%d, validation size:%d, test size:%d" % (
    nb_classes, 3 * a_part, a_part, len(fulltrainset) - 4 * a_part))
    ########################################################################
    # Let us show some of the training images, for fun.

    mpl.use('Agg')  # allows ploting in image without X-server (in docker container)

    # functions to show an image
    def imshow_s(img, name='output_test_cnnhundred.png'):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(name)

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow_s(torchvision.utils.make_grid(images))
    # print labels
    print(' .join:'.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # 2. Define a Convolution Neural Network
    class NetMLP(nn.Module):
        def __init__(self, hiddencells=100):
            super(NetMLP, self).__init__()
            self.fc1 = nn.Linear(32 * 32, hiddencells)
            self.fc2 = nn.Linear(hiddencells, nb_classes)

        def forward(self, x):
            x = x.view(-1, 32 * 32)
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1)
            return x

    class CNN_class(nn.Module):
        def __init__(self, nHidden=256, filter=128, kernerls=5, dropput=0):
            self.nHidden = nHidden
            self.filter = filter
            self.kernerls = kernerls
            self.dropout = dropput

            super(CNN_class, self).__init__()
            self.conv1 = nn.Conv2d(1, self.filter, self.kernerls)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(128, self.filter, self.kernerls)
            self.fc1 = nn.Linear(self.filter * 5 * 5, self.nHidden, nn.Dropout(self.dropout))
            self.fc2 = nn.Linear(self.nHidden, 101)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, self.filter * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1)
            return x
    ########################################################################
    # Define the network to use :
    # net = NetMLP(100)
    net = CNN_class()
    net.to(device)  # move it to GPU or CPU
    # show the structure :
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    ########################################################################
    # Train the network
    # Definition of arrays to store the results and draw the learning curves
    val_err_array = np.array([])
    train_err_array = np.array([])
    nb_sample_array = np.array([])
    best_model = copy.deepcopy(net)

    nb_used_sample = 0
    running_loss = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # print('inputs,labels:',inputs,labels)
            inputs, labels = inputs.to(device), labels.to(device)  # if possible, move them to GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # count how many samples have been used during the training
            nb_used_sample += minibatchsize
            # print/save statistics
            running_loss += loss.item()
            if nb_used_sample % (1000 * minibatchsize) == 0:  # print every 1000 mini-batches
                train_err = (running_loss / (1000 * minibatchsize))
                print('Epoch %d batch %5d ' % (epoch + 1, i + 1))
                print('Train loss : %.3f' % train_err)
                print()
                running_loss = 0.0
                # evaluation on validation set
                totalValLoss = 0.0
                with torch.no_grad():
                    for data in validationloader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        totalValLoss += loss.item()
                val_err = (totalValLoss / len(validationset))
                print('Validation loss mean : %.3f' % val_err)
                print('###################################################')
                train_err_array = np.append(train_err_array, train_err)
                val_err_array = np.append(val_err_array, val_err)
                nb_sample_array = np.append(nb_sample_array, nb_used_sample)
                # save the model only when loss is better
                best_model = copy.deepcopy(net)
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print('Finished Training')

    ### save the best model :##############################################################################
    torch.save(best_model.state_dict(), save_model)

    # Test the network on the test data
    # first on few sample, just to see real results
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    plt.clf()
    imshow_s(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_size)))
    # activate the net with these examples
    outputs = best_model(images)

    # get the maximum class number for each sample, but print the corresponding class name
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(test_size)))

    # Test now  on the whole test dataset.
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    # Check the results for each class
    class_correct = list(0. for i in range(nb_classes))
    class_total = list(0. for i in range(nb_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = best_model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(test_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(nb_classes):
        if class_total[i] > 0:
            print('Accuracy of %5s : %2d %% (%d/%d)' % (
                classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
        else:
            print('No %5s sample' % (classes[i]))


if __name__ == '__main__':
    main()

