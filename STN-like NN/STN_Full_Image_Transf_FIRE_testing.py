"""
This script takes a trained NN model and tests it on the test set provided by the user.
It also visualizes the results of the NN by applying it on the test images.
"""

from torchsummary import summary
import torch
import numpy as np
import glob
import sys
from torch import nn
import time
import cv2
import re
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# Define batch size and epochs
Batch_size = 64
PatchN = 8

# Create a customized dataset class in pytorch
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Image_Path = 'C:/Users/simos/Desktop/FIRE/test_images/'
Trans_Path = 'C:/Users/simos/Desktop/FIRE/test_images/testMatrices1024.txt'
Model_Path = 'C:/Users/simos/Desktop/FIRE/Models/3layer_val.pt'

def imageloader(image_path, trans_path):
    # Read the affine matrices from txt file
    kps = []
    i = 0
    f = open(trans_path, "r")
    for x in f:
        kps.insert(i, re.findall(r"[-+]?\d*\.\d+|\d+", x))
        i += 1
    f.close()

    # Get the filenames of dataset
    filename = glob.glob(image_path + '*.jpg')
    filename.sort()

    images_data = []
    patches_data = []
    targets_data = []
    points_data = []
    names = []
    # Process our images
    j = 0
    for i in range(0, len(filename), 2):
        # Read the first (fixed) image
        fixed = cv2.imread(filename[i])
        fixed = fixed[95:2817, 95:2817]
        fixed = cv2.resize(fixed, (1024, 1024))
        grayfixed = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)
        # Read the second (moved) image
        moving = cv2.imread(filename[i + 1])
        moving = moving[95:2817, 95:2817]
        moving = cv2.resize(moving, (1024, 1024))
        graymoving = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)

        temp_matrix = []
        affine_matrix = []

        m = 0
        for k in range(j, j + 3):
            for l in range(3):
                temp_matrix.insert(m, kps[k][l])
                m += 1
        j += 3

        # Do some data and matrix editing and pass the final form of the affine matrix
        affine_matrix.insert(0, temp_matrix[0:3])
        affine_matrix.insert(1, temp_matrix[3:6])
        # affine_matrix.insert(2, temp_matrix[6:9])
        affine_matrix = np.float32(np.array(affine_matrix))

        if abs(affine_matrix[0][2]) < 220. and abs(affine_matrix[1][2]) < 220.:
            for c in range(0, 8):
                # Sample a 256x256 piece from the resized image
                window_size = 128
                min_x = min_y = 0 + window_size
                max_x = max_y = 1023 - window_size

                # Sample random patches from each test image and save those patch locations for the testing later.
                # Also get and return the whole images, the targets, the names and the top left corner of the sampled
                # patches. This code is for 8 patches per image.
                while True:
                    randomint_x = random.randint(min_x, max_x)
                    randomint_y = random.randint(min_y, max_y)

                    center_point_x = randomint_x
                    center_point_y = randomint_y
                    left = center_point_y - window_size
                    top = center_point_x - window_size
                    right = center_point_y + window_size
                    bottom = center_point_x + window_size

                    moving_image_piece = graymoving[top:bottom, left:right]
                    fixed_image_piece = grayfixed[top:bottom, left:right]
                    threshold = pow((window_size * 2), 2) - 10
                    count = np.count_nonzero

                    # Check if there are more than 10 black pixels in the sampled patch, if yes sample again
                    if count(fixed_image_piece) > threshold and count(moving_image_piece) > threshold:
                        training_images = np.dstack((fixed_image_piece, moving_image_piece))
                        patches_data.append(training_images)
                        points_data.append(np.dstack((left, top)))
                        if c == 7:
                            training_images = np.dstack((grayfixed, graymoving))
                            images_data.append(training_images)
                            targets_data.append(affine_matrix)
                            names.append(filename[i])
                        break

            print('Progress: {:.2f}%'.format((i / len(filename)) * 100), end="\r")
    print('Progress: 100%', end='\r')

    return images_data, patches_data, targets_data, points_data, names


def tester(model, testimageloader, testpatchloader, testtargetloader, testpointsloader, nameloader, criterion):
    since = time.time()

    model.eval()  # Set model to evaluate mode

    patch_loader = testpatchloader
    image_iter = iter(testimageloader)
    target_iter = iter(testtargetloader)
    pts_iter = iter(testpointsloader)
    name_iter = iter(nameloader)

    for i, patches in enumerate(patch_loader, 0):
        targets = next(target_iter)
        images = next(image_iter)
        pts = next(pts_iter).squeeze().to(Device)
        names = next(name_iter)
        images = torch.div(images.to(torch.float32), 255.)
        patches = torch.div(patches.to(torch.float32), 255.)
        patches = patches.to(Device)  # Send data to gpu
        patches = patches.permute(0, 3, 1, 2)
        targets = targets.view(-1, 6)
        targets = targets.to(Device)

        with torch.set_grad_enabled(False):
            output1, output2, output3 = model(patches)  # Test network

            transl_targets = torch.cat((targets.select(1, 2), targets.select(1, 5)), dim=0)
            scale_targets = torch.cat((targets.select(1, 0), targets.select(1, 4)), dim=0)
            rot_targets = torch.cat((targets.select(1, 1), targets.select(1, 3)), dim=0)

            transl_targets = torch.transpose(transl_targets.view(-1, len(targets // 2)), 0, 1)
            scale_targets = torch.transpose(scale_targets.view(-1, len(targets // 2)), 0, 1)
            rot_targets = torch.transpose(rot_targets.view(-1, len(targets // 2)), 0, 1)

            e = 0
            for r in range(0, len(patches), 8):
                # Change the transformation matrix translation, from Local to Global
                output1[r:(r + 8), 0] = - output2[r:(r + 8), 0] * pts[r:(r + 8), 0] - output3[r:(r + 8), 0
                                                                                      ] * pts[r:(r + 8), 1
                                                                                          ] + output1[r:(r + 8), 0
                                                                                              ] + pts[r:(r + 8), 0]
                output1[r:(r + 8), 1] = - output3[r:(r + 8), 1] * pts[r:(r + 8), 0] - output2[r:(r + 8), 1
                                                                                      ] * pts[r:(r + 8), 1
                                                                                          ] + output1[r:(r + 8), 1
                                                                                              ] + pts[r:(r + 8), 1]
                # Pass the Affine transformation variables
                trans_x = torch.mean(output1[r:(r + 8), 0])
                trans_y = torch.mean(output1[r:(r + 8), 1])
                sca_0 = torch.mean(output2[r:(r + 8), 0])
                sca_1 = torch.mean(output2[r:(r + 8), 1])
                rot_0 = torch.mean(output3[r:(r + 8), 0])
                rot_1 = torch.mean(output3[r:(r + 8), 1])

                # Get losses of the predicted transformations with the target transformations
                loss1 = criterion(torch.tensor((trans_x, trans_y), dtype=torch.float32, device='cuda'),
                                  transl_targets[e])
                loss2 = criterion(torch.tensor((sca_0, sca_1), dtype=torch.float32, device='cuda'), scale_targets[e])
                loss3 = criterion(torch.tensor((rot_0, sca_1), dtype=torch.float32, device='cuda'), rot_targets[e])

                # Print results and losses
                print("name: {}".format(names[e]))
                print("output: \n{:.5f} {:.5f} {:.5f} \n{:.5f} {:.5f} {:.5f}".format(sca_0.item(),
                                                                                     rot_0.item(), trans_x.item(),
                                                                                     rot_1.item(), sca_1.item(),
                                                                                     trans_y.item()))
                print("targets: \n{:.5f} {:.5f} {:.5f} \n{:.5f} {:.5f} {:.5f}".format(scale_targets[e][0].item(),
                                                                                      rot_targets[e][0].item(),
                                                                                      transl_targets[e][0].item(),
                                                                                      rot_targets[e][1].item(),
                                                                                      scale_targets[e][1].item(),
                                                                                      transl_targets[e][1].item()))
                print("losses: \ntr:{:.5f} sc:{:.8f} ro:{:.8f}\n\n".format(loss1.item(), loss2.item(), loss3.item()))

                # Show the transformations, applied on the test image set
                T = np.array([[sca_0.cpu(), rot_0.cpu(), trans_x.cpu()],
                              [rot_1.cpu(), sca_1.cpu(), trans_y.cpu()]])
                x = cv2.warpAffine(images.select(3, 1)[e].cpu().numpy(), T, [1024, 1024])

                cv2.imshow('transformed', x)
                cv2.imshow('fixed', images.select(3, 0)[e].cpu().numpy())
                cv2.imshow('moving', images.select(3, 1)[e].cpu().numpy())
                container = np.concatenate((x, images.select(3, 0)[e].cpu().numpy(), images.select(3, 1)[e].cpu().numpy()), axis=1)
                cv2.imshow('transformed-fixed-moving', container)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                e += 1

    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return 0


Test_Images, Test_Patches, Test_Targets, Test_Points, Names = imageloader(Image_Path, Trans_Path)

# Create the dataloaders for the training and testing processes
TestImageLoader = torch.utils.data.DataLoader(Test_Images, batch_size=8, shuffle=False, num_workers=0)
TestPatchLoader = torch.utils.data.DataLoader(Test_Patches, batch_size=Batch_size, shuffle=False, num_workers=0)
TestTargetLoader = torch.utils.data.DataLoader(Test_Targets, batch_size=8, shuffle=False, num_workers=0)
TestPointsLoader = torch.utils.data.DataLoader(Test_Points, batch_size=Batch_size, shuffle=False, num_workers=0)
NameLoader = torch.utils.data.DataLoader(Names, batch_size=8, shuffle=False, num_workers=0)

del Test_Images, Test_Patches, Test_Targets, Test_Points, Names

# Define loss function and optimizer
Model = torch.load(Model_Path)
Model = Model.to(Device)

Criterion = nn.MSELoss()
summary(Model, (2, 256, 256))  # Show model sum-up
tester(Model, TestImageLoader, TestPatchLoader, TestTargetLoader, TestPointsLoader, NameLoader, Criterion)
