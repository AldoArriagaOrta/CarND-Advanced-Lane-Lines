import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use RGB2GRAY if you read an image with mpimg


def cal_undistort(img):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Use cv2.calibrateCamera() and cv2.undistort()
    undist = np.copy(dst)  # Delete this line
    return undist


def color_and_gradient(img, s_thresh = (190, 255), h_thresh = (22, 30), sx_thresh = (25, 80)):
    img = np.copy(img)
    # Convert to HLS color space and split L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    color_binary = cv2.bitwise_or(s_binary, h_binary)
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # three_color_binary = np.dstack((sxbinary, h_binary, s_binary)) * 255
    binary = cv2.bitwise_or(color_binary, sxbinary)
    return binary


def warp(img, inv=False):
    # Perspective transforms, if inv is true returns an unwarped image
    img_size = (img.shape[1], img.shape[0])
    # source coordinates. Do not try to go too "deep" into the perspective, otherwise the warped image is too distorted
    # src = np.float32([[200,720], [1100,720], [620,430], [650,430]])
    # dst = np.float32([[200, 720], [1100, 720], [200, 0], [1100, 0]])
    #
    # src = np.float32([(200, 720), (548, 480), (740, 480), (1110, 720)])
    # dst = np.float32([(200, 720), (200, 0), (1110, 0), (1110, 720)])

    src = np.float32(
        [[((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] / 2) - 90, img_size[1] / 2 + 120],
         [(img_size[0] / 2 + 90), img_size[1] / 2 + 120],
         [(img_size[0] * 5 / 6) + 40, img_size[1]]])

    dst = np.float32(
        [[((img_size[0] / 6) - 10), img_size[1]],
         [((img_size[0] / 6) - 10), 0],
         [(img_size[0] * 5 / 6) + 40, 0],
         [(img_size[0] * 5 / 6) + 40, img_size[1]]])

    if inv == True:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def sliding_window_search(img):
    # This method takes a warped binary image called "img"

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peaks on the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(int(histogram.shape[0] / 2))
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(out_img, np.int_(pts_left), isClosed=False, color=(255, 255, 0), thickness=3)
    cv2.polylines(out_img, np.int_(pts_right), isClosed=False, color=(255, 255, 0), thickness=3)

    return left_fit, right_fit, out_img


def polynomial_search(img, left_fit, right_fit):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(out_img, np.int_(pts_left), isClosed=False, color=(255, 255, 0), thickness=3)
    cv2.polylines(out_img, np.int_(pts_right), isClosed=False, color=(255, 255, 0), thickness=3)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return left_fit, right_fit, result


def calc_x_fit(height, fit):
    # returns the points corresponding to a second order polynomial
    # Takes the image height and fitted polynomial coefficients as input
    y = np.linspace(0, height - 1, height)
    x = fit[0] * y ** 2 + fit[1] * y + fit[2]
    return y, x


def curvature(height, fit):
    # Returns the radius of curvature
    # Takes picture height and fitted polynomial coefficients as input
    y, x = calc_x_fit(height, fit)
    y_eval = y[-1]
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad


def offset(img_height, img_width, left_fit, right_fit):
    y_eval = img_height - 1
    left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

    # left x and right x
    car_center = (left_x + right_x) / 2
    lane_center = img_width / 2
    offset_distance = np.abs(car_center - lane_center)
    offset_distance = np.round(offset_distance * xm_per_pix, 3)
    return offset_distance


def draw_roi(undist,warped, lane):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    img_height = undist.shape[0]
    left_fit, right_fit = lane.get_fit()
    print(left_fit)
    ploty, left_fitx = calc_x_fit(img_height, left_fit)
    ploty, right_fitx = calc_x_fit(img_height, right_fit)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int_(pts_left), isClosed=False, color=(255, 0, 0), thickness=30)
    cv2.polylines(color_warp, np.int_(pts_right), isClosed=False, color=(0, 0, 255), thickness=30)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, inv=True)
    # Combine the result with the original image
    undist_roi = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    left_curverad, right_curverad, curvature = lane.get_curvatures()
    offset = lane.get_offset()

    # left_curv_str = 'left curvature: ' + str(left_curverad) + 'm'
    left_curv_str = 'left curvature: %.2f [m]' %left_curverad
    right_curv_str = 'right curvature: %.2f [m]' %right_curverad
    curv_str = 'curvature: %.2f [m]' %curvature
    offset_str = 'vehicle offset dist: %.2f [m]' %offset

    # Annotate image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    white = (255, 255, 255)
    thickness = 2
    cv2.putText(undist_roi, left_curv_str, (30, 60), font, scale, white, thickness)
    cv2.putText(undist_roi, right_curv_str, (30, 90), font, scale, white, thickness)
    cv2.putText(undist_roi, curv_str, (30, 120), font, scale, white, thickness)
    cv2.putText(undist_roi, offset_str, (30, 150), font, scale, white, thickness)

    return undist_roi


class Lane():
    def __init__(self):
        self.img_width = None
        self.img_height = None
        # was the line detected in the last iteration?
        self.detected = False
        # number of iterations for average filtering
        self.n = 12
        # polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        # polynomial coefficients for the most recent fit
        self.current_fit_left = []
        self.current_fit_right = []
        # radius of curvature of the line in some units
        self.curvature_left = None
        self.curvature_right = None
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None

    def get_fit(self):
        return self.best_fit_left, self.best_fit_right

    def get_curvatures(self):
        return self.curvature_left, self.curvature_right, self.radius_of_curvature

    def get_offset(self):
        return self.line_base_pos

    def sanity_checks(self, left_fit, right_fit):
        _, left_fitx = calc_x_fit(self.img_height, left_fit)
        _, right_fitx = calc_x_fit(self.img_height, right_fit)
        # Check that lines from previous frame have similar curvature

        curv_diff = 10
        left_curverad = curvature(self.img_height, left_fit)
        right_curverad = curvature(self.img_height, right_fit)

        ratio = left_curverad / right_curverad
        if ((ratio >= curv_diff) | (ratio <= 1. / curv_diff)):
            print("Curvature error")
            return False

        # Check that lines are are roughly parallel
        lines_dev = 60
        act_dist = right_fitx - left_fitx
        stddev = np.std(act_dist)
        # print("std dev", dev)
        if (stddev >= lines_dev):
            print("Parallelism error")
            return False

        # Check that lines are separated by a reasonable distance
        line_dist = 100
        nominal_dist = 850  # The distance between lane markings is about 850 pixels
        current_dist = right_fitx[-1] - left_fitx[-1]

        if (np.abs(current_dist - nominal_dist) > line_dist):
            print("Distance error")
            return False

        return True

    def find_lane(self, warpedb_img):
        # input is assumed to be a perspective transformed binary image
        self.img_width = warpedb_img.shape[1]
        self.img_height = warpedb_img.shape[0]

        if (self.detected):
            last_fit_left = self.current_fit_left[-1]
            last_fit_right = self.current_fit_right[-1]
            left_fit, right_fit, lines_img = polynomial_search(warpedb_img, last_fit_left, last_fit_right)
        else:
            left_fit, right_fit, lines_img = sliding_window_search(warpedb_img)

        if (self.sanity_checks(left_fit, right_fit)):
            # pass
            self.detected = True
        else:
            # not pass
            self.detected = False
            # use the last one as current fit
            left_fit = self.current_fit_left[-1]
            right_fit = self.current_fit_right[-1]

        self.current_fit_left.append(left_fit)
        self.current_fit_right.append(right_fit)

        # Remove elements with negative index equal to maximum number of iterations (n) until the end of the array [-n:_]
        if (len(self.current_fit_left) > self.n):
            self.current_fit_left = self.current_fit_left[-self.n:]
            self.current_fit_right = self.current_fit_right[-self.n:]

        # Mean value of last n iterations
        self.best_fit_left = np.mean(self.current_fit_left, axis=0) #Careful with the axis selection with np.mean
        self.best_fit_right = np.mean(self.current_fit_right,axis=0)


        self.curvature_left = curvature(self.img_height, left_fit)
        self.curvature_right = curvature(self.img_height, right_fit)
        self.radius_of_curvature = (self.curvature_left + self.curvature_right) / 2.0  # Average curvature of both lines

        self.line_base_pos = offset(self.img_height, self.img_width, left_fit, right_fit)
        return lines_img

ego_lane = Lane()

def pipeline(image):
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    threshold = color_and_gradient(undistorted)
    warped_binary = warp(threshold)
    ego_lane.find_lane(warped_binary)
    result_img = draw_roi(undistorted, warped_binary, ego_lane)
    return result_img


def process_image(image):

    result=pipeline(image)

    return result


# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

with open('matrix.pickle', 'rb') as handle:
    mtx = pickle.load(handle)

with open('coeff.pickle', 'rb') as handle:
    dist = pickle.load(handle)

project_video1 = 'project_video_out.mp4'
clip1 = VideoFileClip('project_video.mp4')
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_video1, audio=False)

project_video2 = 'challenge_video_out.mp4'
clip2 = VideoFileClip('challenge_video.mp4')
project_clip = clip2.fl_image(process_image)
project_clip.write_videofile(project_video2, audio=False)

project_video3 = 'harder_challenge_video_out.mp4'
clip3 = VideoFileClip   ('harder_challenge_video.mp4')
project_clip = clip3.fl_image(process_image)
project_clip.write_videofile(project_video3, audio=False)


# Section for figure creation (for writeup) , tuning and experimentation
# This code has no purpose in the final version

# # Read in an image
# img = cv2.imread('test_images/test3.jpg')
# undistorted = cv2.undistort(img, mtx, dist, None, mtx)
# threshold = color_and_gradient(undistorted)
# warped = warp(undistorted)
# fit_left, fit_right, slidew_img = sliding_window_search(warped)
# print(fit_left)
# print(fit_right)
# _, _, poly_img = polynomial_search(warped, fit_left, fit_right)
# img_size = (img.shape[1], img.shape[0])
# src = np.float32(
#     [[((img_size[0] / 6) - 10), img_size[1]],
#      [(img_size[0] / 2) - 90, img_size[1] / 2 + 120],
#      [(img_size[0] / 2 + 90), img_size[1] / 2 + 120],
#      [(img_size[0] * 5 / 6) + 40, img_size[1]]])
#
# dst = np.float32(
#     [[((img_size[0] / 6) - 10), img_size[1]],
#      [((img_size[0] / 6) - 10), 0],
#      [(img_size[0] * 5 / 6) + 40, 0],
#      [(img_size[0] * 5 / 6) + 40, img_size[1]]])
# undistorted1 = cv2.polylines(undistorted,np.int32([src]),True,(0,0,255),4)
# warped1 = cv2.polylines(warped,np.int32([dst]),True,(0,0,255),5)
#
# print (src)
# print (dst)
#unwarped = warp(warped, inv=True)

#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(cv2.cvtColor(undistorted1, cv2.COLOR_BGR2RGB))
# ax1.set_title('Undistorted Image', fontsize=12)
# # ax1.axis('off')
# ax2.imshow(cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB))
# ax2.set_title('Warped Image', fontsize=12)
# # ax2.axis('off')
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
#
# output = pipeline (img)
# plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
# plt.show()
