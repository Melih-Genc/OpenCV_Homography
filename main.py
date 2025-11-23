import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Görselleri okutma
src = cv.imread("Görsel1.png")
src2 = cv.imread("Görsel2.png")

#Görselleri Grileştirme
gri_gorsel1 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gri_gorsel2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)

#RGB işlemi
orb = cv.ORB_create()

gri_gorsel1Keypoints, gri_gorsel1Descriptors = orb.detectAndCompute(gri_gorsel1, None)
gri_gorsel2Keypoints, gri_gorsel2Descriptors = orb.detectAndCompute(gri_gorsel2, None)

#Eşleştiriciler
# Using BFMatcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(gri_gorsel1Descriptors, gri_gorsel2Descriptors)
matches = sorted(matches, key=lambda x: x.distance)

#RANSAC
gri_gorsel1_pts = np.float32([gri_gorsel1Keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
gri_gorsel2_pts = np.float32([gri_gorsel2Keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv.findHomography(gri_gorsel1_pts, gri_gorsel2_pts, cv.RANSAC, 5.0)

#yeniden boyutlandırma ve renklendirme
h, w, _ = src2.shape
dsize = (w, h)
img_aligned = cv.warpPerspective(src, H, dsize)
diff_img = cv.absdiff(img_aligned, src2)
diff_gray = cv.cvtColor(diff_img, cv.COLOR_BGR2GRAY)
thresh_value = 10
_, diff_thresh = cv.threshold(diff_gray, thresh_value, 255, cv.THRESH_BINARY)


#Görselleştirme
cv.imshow("1. Hizalanmis Gorsel (Gorsel1)", img_aligned)
cv.imshow("2. Hedef Gorsel (Gorsel2)", src2)
cv.imshow("3. Fark Sonucu (Mutlak Deger)", diff_img)
cv.imshow("4. Fark Sonucu (Eşiklenmis)", diff_thresh)

cv.waitKey(0)
cv.destroyAllWindows()

# ORB Eşleşmelerini de göstermeye eder
final_match_img = cv.drawMatches(gri_gorsel1, gri_gorsel1Keypoints,
                             gri_gorsel2, gri_gorsel2Keypoints, matches[:30], None)
final_match_img = cv.resize(final_match_img, (1000, 650))

plt.figure(figsize=(12, 8))
plt.imshow(cv.cvtColor(final_match_img, cv.COLOR_BGR2RGB))
plt.title("RANSAC Öncesi İlk 30 Eşleşme")
plt.axis('off')
plt.show()
