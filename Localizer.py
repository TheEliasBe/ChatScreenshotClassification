from pytesseract import Output
import pytesseract
import cv2
import pandas as pd
from TextBubbleDetector import Detector, is_in_bounding_box
import imutils
from skimage import io
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import numpy as np

f = ["media/convo/0tkirgfqmcc31.png", "media/convo/0q3roi8vo3j41.jpg", "media/convo/0kby0qwor6h21.jpg", "media/convo/1yavx48bpe621.png"]
images = cv2.imread(f[1])

resized = imutils.resize(images, width=300)
ratio = images.shape[0] / float(resized.shape[0])

rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
height, width, _ = np.shape(rgb)
height, width = height*2, width*2
rgb = cv2.resize(rgb, (int(width),int(height)))
grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

results = pytesseract.image_to_data(grey, output_type=Output.DICT)

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

	if conf > 50 and text != " ":
		# print(text, "Y", y, "ΔY", abs(curr_y - y))
		total_width += w
		cv2.rectangle(rgb, (x,y), (x+w, y+h), (0,0,255),2)
		word_df = word_df.append({'text': text, 'x': x, 'y': y, 'width': w, 'height': h}, ignore_index=True)

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
bounding_boxes['partner'] = bounding_boxes['partner'].astype('int')

# find bounding rectangle of each contour
for c in cnts:
	x, y, w, h = cv2.boundingRect(c)
	bounding_boxes = bounding_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': ''}, ignore_index=True)


# assign each text box to a bounding box
left_most = bounding_boxes['x']
right_most = bounding_boxes['x'] + bounding_boxes['w']

# remove bounding box covering the entire image
bounding_boxes = bounding_boxes[bounding_boxes['w'] != width]


for i2, b in bounding_boxes.iterrows():
	for i1, t in word_df.iterrows():
		box_dim = {'x': int(b['x']), 'y': int(b['y']), 'w': int(b['w']), 'h': int(b['h'])}
		text_dim = {'x': t['x'], 'y': t['y'], 'w': t['width'], 'h': t['height']}
		cv2.rectangle(rgb, (box_dim['x'], box_dim['y']), (box_dim['x']+box_dim['w'], box_dim['y']+box_dim['h']), (255,0,0), 4)
		cv2.rectangle(rgb, (text_dim['x'], text_dim['y']), (text_dim['x'] + text_dim['w'], text_dim['y'] + text_dim['h']), (0, 255, 0), 4)
		if is_in_bounding_box(box_dim, text_dim):
			#print(box_dim)
			#print(text_dim)
			bounding_boxes.at[i2, 'text'] = str(bounding_boxes.at[i2, 'text']) + " " +str(t['text'])

# assign bounding boxes to chat sides
tolerance_side_assignement = 0.17
for i, b in bounding_boxes.iterrows():
	if b['x'] <= tolerance_side_assignement * width:
		bounding_boxes.at[i, 'partner'] = 0
	elif b['x'] + b['w'] >= (1-tolerance_side_assignement)*width:
		bounding_boxes.at[i, 'partner'] = 1
	else:
		bounding_boxes.at[i, 'partner'] = -1

cv2.imwrite('image.jpg', rgb)

# strip text from additional white spaces
bounding_boxes['text'] = bounding_boxes['text'].apply(lambda t : t.strip())

# filter out common stops words in messaging services
stop_words = ['Type', 'a', 'message', 'Type a message', 'Sent', 'sent']
bounding_boxes = bounding_boxes[~bounding_boxes['text'].isin(['Type', 'a', 'message', 'Type a message', 'Sent', 'sent'])]

print(bounding_boxes[(bounding_boxes['text'].str.len() > 0) & (bounding_boxes['partner'] >= 0)])