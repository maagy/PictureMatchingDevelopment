import numpy as np
import cv2
from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_mse as mse
from skimage import data
from skimage.transform import rescale, downscale_local_mean

img1 = cv2.imread(r"j:\tmp\picture.jpg")
img2 = cv2.imread(r"j:\tmp\picture2.jpg")
img1Resized = cv2.resize(img1, (800, 800), 0, 0, interpolation=cv2.INTER_CUBIC)
img2Resized = cv2.resize(img2, (800, 800), 0, 0, interpolation=cv2.INTER_CUBIC)

# cv2.imshow("resized", img1Resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray1 = cv2.cvtColor(img1Resized, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2Resized, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()

kp1 = orb.detect(gray1, None)
kp2 = orb.detect(gray2, None)

kp1, des1 = orb.compute(gray1, kp1)
kp2, des2 = orb.compute(gray2, kp2)

validKeypointsCount = 10
featureAreasX = 6
featureAreasY = 6

# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
img1kp = cv2.drawKeypoints(img1Resized, kp1, None, color=(0, 255, 0), flags=0)
img1kp = cv2.drawKeypoints(img1Resized, kp1[:validKeypointsCount], None, color=(0, 255, 0), flags=0)
img2kp = cv2.drawKeypoints(img2Resized, kp2, None, color=(0, 255, 0), flags=0)
img2kp = cv2.drawKeypoints(img2Resized, kp2[:validKeypointsCount], None, color=(0, 255, 0), flags=0)

height1, width1 = img1.shape[:2]
widthDenominator1 = width1/featureAreasX
heightDenominator1 = height1/featureAreasY

height2, width2 = img2.shape[:2]
widthDenominator2 = width2/featureAreasX
heightDenominator2 = height2/featureAreasY

s = ssim(img1Resized, img2Resized, multichannel=True)
print('similarity: ');
print(s)

nrmse_value = nrmse(img1Resized, img2Resized, norm_type='Euclidean')
print(nrmse_value)
mse_value = mse(img1Resized, img2Resized)
print(mse_value)

# vzit treba 8 nejhorsich podle kp.size + 8 nejlepsich

# kpa1 = map(lambda x: (np.floor(x.pt[0]/widthDenominator1), np.floor(x.pt[1]/heightDenominator1)), kp1)
for kp in kp1:
    print('{} - {}'.format(kp.size, [np.floor(kp.pt[0]/widthDenominator1), np.floor(kp.pt[1]/heightDenominator1)]))

# create BFMatcher object
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = matcher.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

good = np.array([])
# Draw good matches.
goodMatches = filter(lambda x: x.distance < 20.0, matches)
for match in goodMatches:
        print('{} - {} vs {} - {}'.format(match.distance,
                                          [np.floor(kp1[match.queryIdx].pt[0]/widthDenominator1),
                                           np.floor(kp1[match.queryIdx].pt[1]/heightDenominator1)],
                                          [np.floor(kp2[match.trainIdx].pt[0]/widthDenominator2),
                                           np.floor(kp2[match.trainIdx].pt[1]/heightDenominator2)],
                                          np.sqrt(np.sum(np.square(np.subtract(kp1[match.queryIdx].pt,kp2[match.trainIdx].pt))))))


imgMatches = cv2.drawMatches(img1Resized, kp1, img2Resized, kp2, matches[:40], good, flags=2)

cv2.imshow("img1kp", img1kp)
cv2.imshow("img2kp", img2kp)
cv2.imshow("imgMatches", imgMatches)

cv2.waitKey(0)
cv2.destroyAllWindows()

# dst = cv2.cornerHarris(gray,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
