import glob
import cv2
import re
import numpy as np
import random


def generator(base_path, trans_path):
    # Read the affine matrices from txt file
    kps = []
    i = 0
    f = open(trans_path, "r")
    for x in f:
        kps.insert(i, re.findall(r"[-+]?\d*\.\d+|\d+", x))
        i += 1
    f.close()

    # Get the filenames of dataset
    filename = glob.glob(base_path + 'Images/' + '*.jpg')
    filename.sort()
    # Prepare food_path names
    new_path = base_path + 'FIRE_STN_Gray_256_Affine_Food_L/'
    name_len = len(filename[0])
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

        # Get the green channel from the RGB image
        # greenfixed = fixed[:, :, 1]
        # greenfixed = greenfixed / 255.
        # greenfixed = greenfixed[33:991, 33:991]

        # greenmoving = moving[:, :, 1]
        # greenmoving = greenmoving / 255.
        # greenmoving = greenmoving[33:991, 33:991]

        # Get each affine matrix respectively to each image pair
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

        # Check the magnitude of the translation, if it's greater than a threshold
        # do not use that image for sampling
        if abs(affine_matrix[0][2]) < 220. and abs(affine_matrix[1][2]) < 220.:
            # Sample 1000 patches from the selected image
            for c in range(0, 1000):
                # Sample a 256x256 piece from the resized image
                window_size = 128
                min_x = min_y = 0 + window_size
                max_x = max_y = 1023 - window_size
                matrix = np.array(affine_matrix)

                # Check if the sample you got is in the frame
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
                    # And if it contains a lot of black pixels
                    if count(fixed_image_piece) > threshold and count(moving_image_piece) > threshold:
                        training_images = np.dstack((fixed_image_piece, moving_image_piece))
                        # Save the network food into npy files
                        if c // 100 != 0:
                            iname = filename[i][name_len - 9:name_len - 6] + '_%d' % c
                            tname = filename[i][name_len - 9:name_len - 6] + '_%d' % c + '_(T)'
                        elif c // 100 == 0 and c // 10 != 0:
                            iname = filename[i][name_len - 9:name_len - 6] + '_0%d' % c
                            tname = filename[i][name_len - 9:name_len - 6] + '_0%d' % c + '_(T)'
                        else:
                            iname = filename[i][name_len - 9:name_len - 6] + '_00%d' % c
                            tname = filename[i][name_len - 9:name_len - 6] + '_00%d' % c + '_(T)'

                        # If it doesn't, save and move to the next patch
                        np.save(new_path + '%s' % iname, training_images)
                        pt = np.dot(matrix, np.array([[left], [top], [1]], dtype=np.float32))
                        matrix[0, 2] = pt[0] - left
                        matrix[1, 2] = pt[1] - top
                        np.save(new_path + '%s' % tname, np.reshape(matrix, (1, -1)))

                        # cv2.imshow("moving", moving_image_piece)
                        # cv2.imshow("fixed", fixed_image_piece)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        break

        print('Progress: {:.2f}%'.format((i / len(filename)) * 100), end="\r")
    print('Progress: 100%', end='\r')


BasePath = 'C:/Users/simos/Desktop/FIRE/'
TransPath = 'C:/Users/simos/Desktop/FIRE/transMatrices1024.txt'

generator(BasePath, TransPath)
