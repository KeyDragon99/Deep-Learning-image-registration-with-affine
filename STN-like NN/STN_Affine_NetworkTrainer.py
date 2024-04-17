import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torch
import numpy as np
import glob
from STN_Network import NetworkModel
from sklearn.model_selection import train_test_split
import time
import copy
import cv2
import sklearn.utils as ut

# Define batch size and epochs
Batch_size = 128
Epochs = 100
PatchN = 1000
# lr = 0.01
# mmnt = 0.9
# Stairs = 2

# Check if cuda is available and use it
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get paths for dataset and where to save the model after training
Food_Path = 'C:/Users/simos/Desktop/FIRE/FIRE_STN_Gray_256_Affine_L/'
Model_Path = 'C:/Users/simos/Desktop/FIRE/Models/'


# This function loads the dataset files (in saved previously in npy form)
# and passes them as lists to the output. It differentiates the data for training
# and the target output data into two lists.
def datasetloader(folder, patchn):
    filename = glob.glob(folder + '*.npy')
    filename.sort()

    # This class loads the dataset files
    class Loader:
        def __init__(self):
            self.food_piece = np.load(filename[i], allow_pickle=True)

        def __getitem__(self, name):
            return self.food_piece

    # Load the Images and the Targets from the dataset
    images_dataset = []
    targets_dataset = []
    # Get the Images and the Targets from the npy files
    for i in range(0, len(filename)):
        c = len(filename[i]) - 6
        if filename[i][c] != 'T':
            images = Loader().__getitem__(filename[i])
            images_dataset.append(images)
        else:
            targets = Loader().__getitem__(filename[i])
            # for j in range(0, patchn):
            targets_dataset.append(targets)
        print('Loading Dataset: {:.2f}%'.format((i / len(filename)) * 100), end='\r')

    return np.array(images_dataset, dtype=np.uint8), np.array(targets_dataset, dtype=np.float32)


# This is the main function of the training session, it takes the dataloaders containing the images and the targets
# both for training and validation. It also takes the model's function containing its structure and the criterion
# function and the optimizer function. There are two custom-made learning rate schedulers that can be used for SGD
# if wanted.
def trainer(model, criterion, optimizer, trainimageloader, traintargetloader, valimageloader, valtargetloader):
    # global lr
    # global Stairs
    # prev_epo_loss = 0.0

    # Keep track of the training and validation processes
    train_loss_history, val_loss_history = [], []

    # Keep track of time stamps
    since = time.time()
    # Keep track of the best validation pass so far
    best_model_wts = copy.deepcopy(model.state_dict())

    # In this loop we train the network model
    for epoch in range(Epochs):
        since_epoch = time.time()

        # Uncomment in function global variables lr, Stairs and outside global Stairs
        # Keep prev_epo_loss variable in comment
        # if epoch == int(Epochs - Epochs / Stairs) and Stairs != 8:
        #     Stairs = Stairs * 2
        #     for g in optimizer.param_groups:
        #         lr = lr / 10.0
        #         g['lr'] = lr
        #     train_loss_history.append('Learning rate changed to: {}'.format(lr))
        #     val_loss_history.append('Learning rate changed to: {}'.format(lr))

        # Switch between training and validation modes
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                image_loader = trainimageloader
                target_iter = iter(traintargetloader)
            else:
                model.eval()  # Set model to evaluate mode
                image_loader = valimageloader
                target_iter = iter(valtargetloader)

            running_loss = 0.0

            for i, images in enumerate(image_loader, 0):
                targets = next(target_iter)
                targets = targets.to(Device)
                targets = targets[:, None, :]
                images = torch.div(images.to(torch.float32), 255.)
                images = images.to(Device)  # Send data to gpu
                images = images.permute(0, 3, 1, 2)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        output1, output2, output3 = model(images)  # Train network

                        transl_targets = torch.cat((targets.select(2, 2), targets.select(2, 5)), dim=1)
                        scale_targets = torch.cat((targets.select(2, 0), targets.select(2, 4)), dim=1)
                        rot_targets = torch.cat((targets.select(2, 1), targets.select(2, 3)), dim=1)
                        loss1 = criterion(output1, transl_targets)  # Calculate loss
                        loss2 = criterion(output2, scale_targets)
                        loss3 = criterion(output3, rot_targets)
                    else:
                        output1, output2, output3 = model(images)  # Validate network

                        transl_targets = torch.cat((targets.select(2, 2), targets.select(2, 5)), dim=1)
                        scale_targets = torch.cat((targets.select(2, 0), targets.select(2, 4)), dim=1)
                        rot_targets = torch.cat((targets.select(2, 1), targets.select(2, 3)), dim=1)
                        loss1 = criterion(output1, transl_targets)  # Calculate loss
                        loss2 = criterion(output2, scale_targets)
                        loss3 = criterion(output3, rot_targets)

                    # cv2.imshow('1st output', output1[0].detach().cpu().numpy())
                    # cv2.imshow('2nd output', output2[0].detach().cpu().numpy())
                    # cv2.imshow('fixed', images[0, 0].detach().cpu().numpy())
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    if phase == 'train':
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)
                        loss3.backward()
                        optimizer.step()

                running_loss += (float(loss1) + float(loss2) + float(loss3))

            epoch_loss = float(running_loss) / len(image_loader)

            if phase == 'train':
                print('Epoch: {}/{}'.format(epoch + 1, Epochs))
            print('Current {} loss: {}'.format(phase, epoch_loss))

            if phase == 'val' and epoch == 0:
                best_loss = epoch_loss

            # Deep copy the model if this is the best validation loss so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                print('Best val loss: {:.8f}'.format(best_loss))

            # Keep track of things
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_loss_history.append(epoch_loss)

            # Uncomment prev_epo_loss in function if using this lr scheduler, keep Stairs variable in comment
            # if phase == 'val' and epoch != 0 and lr > 0.00001 and epoch_loss > prev_epo_loss:
            #     for g in optimizer.param_groups:
            #         lr = lr/10.0
            #         g['lr'] = lr
            #     train_loss_history.append('Learning rate changed to: {}'.format(lr))
            #     val_loss_history.append('Learning rate changed to: {}'.format(lr))
            # if phase == 'val':
            #     prev_epo_loss = epoch_loss

        # Keep track of time
        time_elapsed = time.time() - since_epoch
        print('Epoch completed in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    # Summarize final time and best validation error in the end
    time_elapsed = time.time() - since
    print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation error: {:8f}'.format(best_loss))

    model.load_state_dict(best_model_wts)

    return model, val_loss_history, train_loss_history, time_elapsed


# This function performs all the back-up operations.
def backup(model, val_history, train_history, timer):
    torch.save(model, Model_Path + '7layer_val.pt')  # Save the network model

    time.sleep(0.5)
    print('Done!')
    print('Backing up collected data..', end='')

    # Save the data and some of the model's properties-hyperparameters
    train_file = open('train_history.txt', 'a')
    val_file = open('val_history.txt', 'a')
    # Save in one file the raw data
    raw_file = open('raw.txt', 'a')

    # Save details for Model structure, variables used and functions used
    val_file.write(
        "Total Epochs: {}\nBatch Size: {}\nDevice: {}\nCriterion: L2-mean\nOptimizer: Adam\n7 Layer Encoder"
        "(4fc, 2x0.2drop, parallel affine)\n(no black, 256, 3-7, separated, Local, shuffle2)\n".
        format(Epochs,
               Batch_size,
               Device))
    train_file.write(
        "Total Epochs: {}\nBatch Size: {}\nDevice: {}\nCriterion: L2-mean\nOptimizer: Adam\n7 Layer Encoder"
        "(4fc, 2x0.2drop,  parallel affine)\n(no black, 256, 3-7, separated, Local, shuffle2)\n"
        "Time: {:.0f}h {:.0f}m {:.0f}s\n".format(Epochs, Batch_size,
                                                 Device,
                                                 timer // 3600,
                                                 (timer // 60) % 60,
                                                 timer % 60))

    i = 0
    for loss in iter(val_history):
        if type(loss) == float:
            val_file.write('Epoch: {} '.format(str(i + 1)) + 'Validation Loss: {:.6f} \n'.format(loss))
            raw_file.write('{:.6f} '.format(loss))
            i = i + 1
        else:
            val_file.write('{}\n'.format(loss))
    i = 0
    for loss in iter(train_history):
        if type(loss) == float:
            train_file.write('Epoch: {} '.format(str(i + 1)) + 'Train Loss: {:.6f} \n'.format(loss))
            i = i + 1
        else:
            train_file.write("{}\n".format(loss))

    raw_file.write('\n')
    val_file.write('\n###############################\n')
    train_file.write('\n###############################\n')

    train_file.close()
    val_file.close()
    raw_file.close()


# This function performs a custom-made data-split, specifically made for the case of splitting the
# dataset containing patches from the same images. This function is splitting the dataset among the different
# images out of which the patches were extracted. If we were to use any other kind of already-made splitter
# function, the splitting would be performed between the image patches, resulting in a biased training.
# The function takes the image patches, the targets the patch number extracted from each image and the train ratio
# as well as if you want to shuffle the dataset or not.
def datasplit(images, targets, patchn, train_ratio, shuffle):
    print('Splitting Dataset..', end='')
    val_ratio = round(1 - train_ratio, 2)
    train_images, val_images, train_targets, val_targets = [], [], [], []
    samples = []

    # Sample the index of the images that will be split
    for i in range(0, len(images) // patchn):
        samples.append(i)

    # Randomly sample and shuffle using the sampled indexes
    train_samples, test_val_samples = train_test_split(samples, test_size=val_ratio, train_size=train_ratio,
                                                       shuffle=shuffle)
    # Do it again for the train-test split of the dataset
    val_samples, test_samples = train_test_split(test_val_samples, test_size=val_ratio, train_size=train_ratio,
                                                 shuffle=shuffle)
    # Print the images selected for each split of the dataset
    print("train {}".format(train_samples))
    print("val {}".format(val_samples))
    print("test {}".format(test_samples))
    time.sleep(60)

    # Using the split and shuffled indexes we got before, pass the image patches accordingly to lists
    for i in range(0, len(train_samples)):
        j = train_samples[i] * patchn
        train_images.append(images[j:j + patchn])
        train_targets.append(targets[j:j + patchn])

    for i in range(0, len(val_samples)):
        j = val_samples[i] * patchn
        val_images.append(images[j:j + patchn])
        val_targets.append(targets[j:j + patchn])

    # Make into numpy array and reshape accordingly to be able to handle the dataset with PyTorch later
    train_images = np.reshape(np.array(train_images), (-1, 256, 256, 2))
    train_targets = np.reshape(np.array(train_targets), (-1, 6))
    val_images = np.reshape(np.array(val_images), (-1, 256, 256, 2))
    val_targets = np.reshape(np.array(val_targets), (-1, 6))
    # Shuffle the data among the patches
    if shuffle:
        train_images, train_targets = ut.shuffle(train_images, train_targets, random_state=0)
        val_images, val_targets = ut.shuffle(val_images, val_targets, random_state=0)

    return train_images, train_targets, val_images, val_targets


# Load dataset files
Images, Targets = datasetloader(Food_Path, PatchN)
print('Loading Dataset..Done!')

# Split dataset
Train_Images, Train_Targets, Val_Images, Val_Targets = datasplit(Images, Targets, PatchN, train_ratio=0.7, shuffle=True)
del Images, Targets
time.sleep(0.5)
print('Done!')

print('Passing Training Images..', end='')
# Create the dataloaders for the training and validation processes
TrainImageLoader = torch.utils.data.DataLoader(Train_Images, batch_size=Batch_size, shuffle=False, num_workers=0)
TrainTargetLoader = torch.utils.data.DataLoader(Train_Targets, batch_size=Batch_size, shuffle=False, num_workers=0)
del Train_Images, Train_Targets
time.sleep(0.5)
print('Done!')

print('Passing Validation Images..', end='')
ValImageLoader = torch.utils.data.DataLoader(Val_Images, batch_size=Batch_size, shuffle=False, num_workers=0)
ValTargetLoader = torch.utils.data.DataLoader(Val_Targets, batch_size=Batch_size, shuffle=False, num_workers=0)
del Val_Images, Val_Targets
time.sleep(0.5)
print('Done!')

print('Initializing Network Model..', end='')
# Load and pass the model to Device
Model = NetworkModel()
# Model = nn.DataParallel(Model) # Use parallel if available
Model.to(Device)
time.sleep(0.5)
print('Done!\n')

# Define loss function and optimizer
Criterion = nn.MSELoss(reduction='mean')
# Criterion = nn.BCELoss(reduction='mean')
# Criterion = nn.L1Loss(reduction='sum')
# Optimizer = optim.SGD(Model.parameters(), lr=lr, momentum=mmnt)
Optimizer = optim.Adam(Model.parameters())

print('Initializing training process..\n')
time.sleep(0.5)
summary(Model, (2, 256, 256))  # Show model sum-up

# Start training
Model, Val_History, Train_History, Timer = trainer(Model, Criterion, Optimizer, TrainImageLoader, TrainTargetLoader,
                                                   ValImageLoader, ValTargetLoader)

# Backup collected data
print('Backing up network...', end='')
backup(Model, Val_History, Train_History, Timer)
time.sleep(0.5)
print('Done!')
