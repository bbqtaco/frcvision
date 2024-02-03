from pyapriltags import Detector
import numpy as np
import cv2
from util import drawCube,  solveExtrinsicsFromH, drawcubeprojection

#def camera_pose_from_homography(Kinv, H):
#    '''Calculate camera pose from Homography.
#
#    Args:
#       Kinv: inverse intrinsic camera matrix
#       H: homography matrix
#    Returns:
#       R: rotation matrix
#       T: translation vector
#    '''
#    H = np.transpose(H)
#    # the scale factor
#    l = 1 / np.linalg.norm(np.dot(Kinv, H[0]))
#    r1 = l * np.dot(Kinv, H[0])
#    r2 = l * np.dot(Kinv, H[1])
#    r3 = np.cross(r1, r2)
#    T = l * np.dot(Kinv, H[2])
#    R = np.array([[r1], [r2], [r3]])
#    R = np.reshape(R, (3, 3))
#    U, S, V = np.linalg.svd(R, full_matrices=True)
#    U = np.matrix(U)
#    V = np.matrix(V)
#    R = U * V
#    return (R, T)

#### need to determine camera params???
#c_xy are center pixels
#f_xy are focal length in pixels
cx = 640.0/2
cy = 480.0 / 2
fx = 1000.0
fy = 1000.0

#K matrix for my own camera, change for your own camera, my camera has no radial distortion effects but they can be added
####k = np.array([[2063.8820747760683, 0, 1184.6568232403756],[0, 2060.440690811184, 763.9115119061984],[0,0,1]])

### get these matrices from calibrate_mycamera.py  (first take a bunch of pictures with the camera of chess board.
k = np.array([[565.24737763,   0.        , 297.49928744],
       [  0.        , 603.62419145, 247.53862091],
       [  0.        ,   0.        ,   1.        ]])
dist_coeffs = np.array([[-7.66438962e-01,  3.95367443e+00, -3.01231845e-02,
        -8.49946006e-03, -1.11860015e+01]])

#dist_coeffs = np.zeros((5, 1), np.float32)



corner_tag_global_frame = np.array([[0,0,0],[0,3,0],[3,3,0],[3,0,0]], dtype = np.float64)
#corner_tag_global_frame = np.flipud(corner_tag_global_frame)
corner_tag_pixel = []


camera_params = (fx,fy,cx, cy)

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    rval = False

while rval:
    #cv2.imshow("preview", grayframe)
    #out = at_detector.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

    #out = at_detector.detect(grayframe)
    out = at_detector.detect(grayframe, estimate_tag_pose=True, camera_params=camera_params, tag_size=1)
    #cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    #out = at_detector.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    print(out)
    if len(out) > 0:
        (ptA, ptB, ptC, ptD) = out[0].corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
    
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (255, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (255, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (255, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (255, 255, 0), 2)
    
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(out[0].center[0]), int(out[0].center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        ## draw vector
        # Here, solvePnP does most of the work for us. It automatically determines
        #that, in our case, the PnP problem is degenerate (due to coplanarity) and
        # reverts to extracting the pose of the camera via a homography estimation.
        #It goes a step further and does a least squared optimization on the rotation matrix to find a
        # truly orthonormal matrix. This can be seen the quality of the box
        #being drawn on the AprilTag in comparison to the unoptimized version
        axis = np.float64([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        ret,rvec,tvec = cv2.solvePnP(corner_tag_global_frame, out[0].corners,k,dist_coeffs)

        ## lower tech way
        #h = out[0].homography
        #rotation,tvec = solveExtrinsicsFromH(h,k)
        #print(rotation)
        #rvec,_ = cv2.Rodrigues(rotation)

        
        
        # We can use the estimated pose to compute the corners of a cube as
        #if it was imaged on the plane of the fiducial itself, using the pin hole camera model
        # [u,v] = K[R|t][X,Y,Z]
        #cubePts,_ = cv2.projectPoints(np.array([[0,0,1],[0,1,1],[1,1,1],[1,0,1]],dtype = np.float64),rvec,tvec,k,dist_coeffs)
        #vecpts,_ = cv2.projectPoints(np.array([cX,cY,1.0]),rvec,tvec,k,dist_coeffs)
        #vecpts,_ = cv2.projectPoints(np.array([cY,cX,10.0]),rvec,tvec,k,dist_coeffs)
        #print("vector point=",vecpts)
        #cv2.line(frame,np.array([cX,cY]),(int(np.squeeze(vecpts)[0]), int(np.squeeze(vecpts)[1])),(255,255,0),2)
        #cv2.line(frame,np.array([cX,cY]),(int(cx),int(cy)),(255,255,0),2)
        cubePts,_ = cv2.projectPoints(axis,rvec,tvec,k,dist_coeffs)
        #frame = drawCube(frame, out[0].corners, cubePts)
        frame = drawcubeprojection(frame, out[0].corners, cubePts)
        centerpos = (int(out[0].center[0]), int(out[0].center[1]))
        label = '{:}'.format(out[0].tag_id)
        cv2.putText(frame,label, centerpos  , cv2.FONT_HERSHEY_SIMPLEX,
                    1, #font size
                    (209, 80, 0, 255), #font color
                    3)
       
        
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()


    



	
	
