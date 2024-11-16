import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/home/nvidia/jetson-inference/data/images/humans_1.jpg")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("my_result.jpg") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	print(detections[0])
	if img is None: # capture timeout
		continue
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
