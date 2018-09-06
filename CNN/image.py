import numpy as np
import cv2
import cnn

while(True):
	print('')
	print('Chose an option:')
	print('a) Train Model')
	print('b) Test Model')
	print('c) Test Model with OpenCV')
	print('e) Exit')
	print('')
	option = input()
	print('')

	if option.lower() == 'c':
		print('========================================')
		print('Press c to take a picture and predict')
		print('Press q to exit')
		print('========================================')
		cap = cv2.VideoCapture(0)	# 0 first camera (laptop camera)
		while(True):
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rows,cols = gray.shape
			lefUpCornerX = int(cols/2-100)
			lefUpCornerY = int(rows/2-100)
			rightDownX = int(cols/2+100)
			rightDownY = int(rows/2+100)
			cv2.rectangle(frame, (lefUpCornerX,lefUpCornerY), (rightDownX,rightDownY), (0,219,255), 4)
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('c'):
				print('Captured!')
				data = gray[lefUpCornerY:rightDownY,lefUpCornerX:rightDownX]
				cv2.imshow('data',data)
				npImg, npLabel = cnn.process_data(data,'A')
				test_data = [[npImg,'Test']]
				model = cnn.create_model()
				print(cnn.predict_data(test_data[0],model))
			elif cv2.waitKey(1) & 0xFF == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()

	elif option.lower() == 'a':
		model = cnn.create_model()

	elif option.lower() == 'b':
		test_data = cnn.process_test_data()
		cnn.run_test_data(test_data,model)

	elif option.lower() == 'e':
		break

	else:
		print('Please select a valid option')
