import numpy as np
import cv2

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def camera_pose_from_homography(Kinv, H):
    '''Calculate camera pose from Homography.

    Args:
       Kinv: inverse intrinsic camera matrix
       H: homography matrix
    Returns:
       R: rotation matrix
       T: translation vector
    '''
    H = np.transpose(H)
    # the scale factor
    l = 1 / np.linalg.norm(np.dot(Kinv, H[0]))
    r1 = l * np.dot(Kinv, H[0])
    r2 = l * np.dot(Kinv, H[1])
    r3 = np.cross(r1, r2)
    T = l * np.dot(Kinv, H[2])
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    U, S, V = np.linalg.svd(R, full_matrices=True)
    U = np.matrix(U)
    V = np.matrix(V)
    R = U * V
    return (R, T)


def drawCube(frame,tag_corners, cube_corners):
    (CubePtA,CubePtB,CubePtC,CubePtD) = cube_corners
    (ptA, ptB, ptC, ptD) = tag_corners
    
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
  
    CubePtA = (int(CubePtA[0][0]), int(CubePtA[0][1]))
    CubePtB = (int(CubePtB[0][0]), int(CubePtB[0][1]))
    CubePtC = (int(CubePtC[0][0]), int(CubePtC[0][1]))
    CubePtD = (int(CubePtD[0][0]), int(CubePtD[0][1]))

    
    cv2.circle(frame, ptA, 20, (0, 0, 255), -1)
    cv2.circle(frame, ptB, 20, (0, 255, 0), -1)
    cv2.circle(frame, ptC, 20, (255, 0, 0), -1)
    cv2.circle(frame, ptD, 20, (255, 255, 255), -1)
    
    cv2.circle(frame, CubePtA, 20, (0, 0, 255), -1)
    cv2.circle(frame, CubePtB, 20, (0, 255, 0), -1)
    cv2.circle(frame, CubePtC, 20, (255, 0, 0), -1)
    cv2.circle(frame, CubePtD, 20, (255, 255, 255), -1)

    cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
    cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
    cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
    cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

    cv2.line(frame, CubePtA, CubePtB, (0, 255, 0), 2)
    cv2.line(frame, CubePtB, CubePtC, (0, 255, 0), 2)
    cv2.line(frame, CubePtC, CubePtD, (0, 255, 0), 2)
    cv2.line(frame, CubePtD, CubePtA, (0, 255, 0), 2)

    cv2.line(frame, CubePtA, ptA, (0, 255, 0), 2)
    cv2.line(frame, CubePtB, ptB, (0, 255, 0), 2)
    cv2.line(frame, CubePtC, ptC, (0, 255, 0), 2)
    cv2.line(frame, CubePtD, ptD, (0, 255, 0), 2)
    
    return frame


def solveExtrinsicsFromH(h,k):
    b = (np.linalg.inv(k)@h)
    
    if np.linalg.det(b)<0:
        b =  b*(-1)
    [a1,a2,a3] = b.T
    
    #Instead of normalizing by the value of A1, or A2, we average the two and normalize by both to create the better rotation
    # matrix
    Lambda = 2/(np.linalg.norm(a1) + np.linalg.norm(a2))
    r1 = Lambda * a1
    r2 = Lambda * a2
    r3 = np.cross(r1, r2, axis = 0)
    t = Lambda * a3
    h =  np.column_stack((r1, r2, r3))
     
    return h,t

def drawcubeprojection(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img
