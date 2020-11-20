#import cv2
#import numpy as np
#from matplotlib import pyplot as plt

#cv2.namedWindow("preview")
#vc = cv2.VideoCapture(0)

#if vc.isOpened(): # try to get the first frame
#    rval, frame = vc.read()
#else:
#    rval = False

#while rval:
#    cv2.imshow("preview", frame)
#    rval, frame = vc.read()
#    key = cv2.waitKey(20)

#    img2 = frame

#    # All the 6 methods for comparison in a list
#    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#    for meth in methods:
#        img = img2.copy()
#        template = cv2.imread('Black_Polo_Headshot.jpg')
#        w, h = template.shape[:-1]
#        #method = eval(meth)

#        method = 0

#        # Apply template Matching
#        res = cv2.matchTemplate(img,template,method)
#        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#            top_left = min_loc
#        else:
#            top_left = max_loc
#        bottom_right = (top_left[0] + w, top_left[1] + h)

#        cv2.rectangle(img,top_left, bottom_right, 255, 2)

#        plt.subplot(121),plt.imshow(res,cmap = 'gray')
#        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#        plt.subplot(122),plt.imshow(img,cmap = 'gray')
#        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#        plt.suptitle(meth)

#        plt.show()


#    if key == 27: # exit on ESC
#        break
#cv2.destroyWindow("preview")

import cv2
from matplotlib import pyplot as plt
import numpy as np
import filter_system as fs



cap = cv2.VideoCapture(0) #Webcam Capture
toss, snap = cap.read()
temp = cv2.imread('test_template_2.jpg',0)
pf = fs.ParticleFilter(snap,temp)

while(True):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#for i in range(3):
		
	#	template = cv2.imread('test_template_2.jpg',0)
	#	w, h = template.shape[::-1]

	#	res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF)

	#	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	#	top_left = min_loc
	#	bottom_right = (top_left[0] + w, top_left[1] + h)

	#	cv2.rectangle(frame,top_left, bottom_right, 255, 1)
	#	cv2.putText(frame, 'Detected Face ID: '+str(i), (top_left[0],top_left[1]-10), 
	#			cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))
	
	template = cv2.imread('test_template_2.jpg',0)
	pf.process(frame)
	pf.render(frame)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	top_left = min_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(frame,top_left, bottom_right, 255, 1)
	cv2.putText(frame, 'Detected Face ID: ', (top_left[0],top_left[1]-10), 
			cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))

	cv2.imshow('Test',frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()		

