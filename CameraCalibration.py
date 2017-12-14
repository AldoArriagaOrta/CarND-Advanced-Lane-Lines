import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import glob

# Load camera images
images = glob.glob('camera_cal/calibration*.jpg')

# prepare object points
nx = 9#number of inside corners in x
ny = 6#number of inside corners in y

# Arrays to store object and image points from calibration images
objpoints = []  # 3 dimensional points in world space
imgpoints = []  # 2 dimensional points in image space

# Prepare objects like (0,0,0), (1,0,0), etc
objp = np.zeros((6 * 9, 3), np.float32)  # 9x6 points, each with 3 columns for XYZ coordinates of each corner
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # map obj coordinates and reshape them into 2 columns to store in XY

for fname in images:
    # Make a list of calibration images
    img = cv2.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If corners are found, add object points and image points to the arrays
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# These lines are used to visualize the result and select the images to be saved for the writeup.
# They can be (un)commented for that goal, though they are not really needed in the final code.

# for fname in images:
#     img = cv2.imread(fname)
#     gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Or use RGB2GRAY if you read an image with mpimg
#     undistorted = cv2.undistort(gray, mtx, dist, None, mtx)
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#     f.tight_layout()
#     ax1.imshow(img)
#     ax1.set_title('Original Image', fontsize=12)
#     ax2.imshow(undistorted, cmap='gray')
#     ax2.set_title('Undistorted Image', fontsize=12)
#     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.show()

# Store data (serialize)
with open('matrix.pickle', 'wb') as handle:
    pickle.dump(mtx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Matrix data stored')

with open('coeff.pickle', 'wb') as handle:
    pickle.dump(dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Distortion coefficient data stored')
