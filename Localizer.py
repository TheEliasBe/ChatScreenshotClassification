from pytesseract import Output
import pytesseract
import cv2
import pandas as pd
from TextBubbleDetector import Detector, is_in_bounding_box
import imutils
from skimage import io
import numpy as np

f = ["media/convo/0tkirgfqmcc31.png", "media/convo/0q3roi8vo3j41.jpg", "media/convo/0kby0qwor6h21.jpg"]
images = cv2.imread(f[0])

resized = imutils.resize(images, width=300)
ratio = images.shape[0] / float(resized.shape[0])

rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
text = pytesseract.image_to_boxes(rgb)

# number of pixels +- still considered aline of text
line_threshold = 10

# number of pixelx +- still considered a text bubble
line_break_threshold = 66

# variables
lines = []
lines_df = pd.DataFrame(columns=['text', 'x', 'y', 'width', 'height'])
word_df = pd.DataFrame(columns=['text', 'x', 'y', 'width', 'height'])
l = []
curr_y = 0
curr_x = 0
total_width = 0
x_list = []
# find lines of text and merge words into lines
for i in range(0, len(results["text"])):
	# extract the bounding box coordinates of the text region from
	# the current result
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]
	# extract the OCR text itself along with the confidence of the
	# text localization
	text = results["text"][i]

	conf = int(results["conf"][i])


	if conf > 10 and text != " ":
		# print(text, "Y", y, "ΔY", abs(curr_y - y))
		total_width += w
		cv2.rectangle(rgb, (x,y), (x+w, y+h), (0,0,255),2)
		word_df = word_df.append({'text': text, 'x': x, 'y': y, 'width': w, 'height': h}, ignore_index=True)
		if abs(curr_y - y) <= line_threshold or curr_y == 0:
			l.append(text)
			x_list.append(x)
		else:
			lines_df = lines_df.append({'text': l.copy(), 'x': min(x_list), 'y': curr_y, 'width': total_width, 'height': h}, ignore_index=True)
			x_list = []
			x_list.append(x)
			# print("Line Finished", l)
			lines.append(l.copy())
			l.clear()
			total_width = 0
			l.append(text)
		curr_y = y

print("word_df")
print(word_df)

# find locations of text bubbles

# grayscale the image, blur it and apply threshold to emphasize egdes
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
thresh = cv2.threshold(blurred, 247, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

bounding_boxes = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'text', 'partner'])
bounding_boxes['text'] = bounding_boxes['text'].astype('string')

# find bounding rectangle of each contour
for c in cnts:
	x, y, w, h = cv2.boundingRect(c)
	bounding_boxes = bounding_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': ''}, ignore_index=True)


# assign each text box to a bounding box
left_most = bounding_boxes['x']
right_most = bounding_boxes['x'] + bounding_boxes['w']

# remove bounding box covering the entire image
h, w, _ = np.shape(rgb)
bounding_boxes = bounding_boxes[bounding_boxes['w'] != w]


for i2, b in bounding_boxes.iterrows():
	for i1, t in word_df.iterrows():
		box_dim = {'x': int(b['x']), 'y': int(b['y']), 'w': int(b['w']), 'h': int(b['h'])}
		text_dim = {'x': t['x'], 'y': t['y'], 'w': t['width'], 'h': t['height']}
		cv2.rectangle(rgb, (box_dim['x'], box_dim['y']), (box_dim['x']+box_dim['w'], box_dim['y']+box_dim['h']), (255,0,0), 2)
		cv2.rectangle(rgb, (text_dim['x'], text_dim['y']), (text_dim['x'] + text_dim['w'], text_dim['y'] + text_dim['h']), (0, 255, 0), 2)
		if is_in_bounding_box(box_dim, text_dim):
			#print(box_dim)
			#print(text_dim)
			bounding_boxes.at[i2, 'text'] = str(bounding_boxes.at[i2, 'text']) + " " +str(t['text'])
print(bounding_boxes)

# assign bounding boxes to chat sides
for i, b in bounding_boxes.iterrows():
	...

cv2.imwrite('image.jpg', rgb)