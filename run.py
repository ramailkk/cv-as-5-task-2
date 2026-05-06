import cv2
cap = cv2.VideoCapture('input_video.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Your video has {total_frames} frames")
cap.release()