import argparse
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb',
                        type=str,
                        help='path to the rgb image')
    parser.add_argument('infrared',
                        type=str,
                        help='path to the infrared image')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.75,
                        help='matching ratio')
    parser.add_argument('--threshold',
                        type=int,
                        default=4,
                        help='RANSAC reprojection threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 读取图像 read image
    img_rgb = cv2.imread(args.rgb)
    img_infrared = cv2.imread(args.infrared)

    # 缩放 rescale
    while img_rgb.shape[0] > 1024 or img_rgb.shape[1] > 1024:
        img_rgb = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    while img_infrared.shape[0] > 1024 or img_infrared.shape[1] > 1024:
        img_infrared = cv2.resize(img_infrared, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    # 检测角点 detect keypoints
    detector = cv2.SIFT_create()
    kps_rgb, des_rgb = detector.detectAndCompute(img_rgb, None)
    kps_infrared, des_infrared = detector.detectAndCompute(img_infrared, None)

    # 匹配角点 match keypoints
    # matcher = cv2.BFMatcher()  # 暴力匹配
    matcher = cv2.FlannBasedMatcher({
        'algorithm': 0,
        'trees': 5,
    }, {
        'checks': 100,
    })
    matches = matcher.knnMatch(des_rgb, des_infrared, k=2)
    matches_good = []
    for m1, m2 in matches:
        if m1.distance < args.ratio * m2.distance:
            matches_good.append([m1])

    # 计算单位性矩阵 compute homography matrix
    pts_rgb = np.float32([kps_rgb[ms[0].queryIdx].pt for ms in matches_good]).reshape(-1, 1, 2)
    pts_infrared = np.float32([kps_infrared[ms[0].trainIdx].pt for ms in matches_good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_rgb, pts_infrared, cv2.RANSAC, args.threshold)

    # 对齐分辨率 align resolution
    img_infrared_registered = cv2.warpPerspective(img_infrared, H, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 展示结果 show results
    cv2.imshow('keypoints on rgb image', cv2.drawKeypoints(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), kps_rgb, None))
    cv2.imshow('keypoints on infrared image', cv2.drawKeypoints(cv2.cvtColor(img_infrared, cv2.COLOR_BGR2GRAY), kps_infrared, None))
    cv2.imshow('registered infrared image', img_infrared_registered)
    cv2.imshow('matches', cv2.drawMatchesKnn(img_rgb, kps_rgb, img_infrared, kps_infrared, matches_good, None, flags=2))

    if cv2.waitKey() & 0xff == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

