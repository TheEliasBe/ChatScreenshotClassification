from pytesseract import Output
import pytesseract
import argparse
import cv2
import pandas as pd

images = cv2.imread("media/convo/0kby0qwor6h21.jpg")
rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT, lang='eng')
text = pytesseract.image_to_boxes(rgb)

# number of pixels +- still considered aline of text
line_threshold = 10

# number of pixelx +- still considered a text bubble
line_break_threshold = 66

# variables
lines = []
lines_df = pd.DataFrame(columns=['text', 'x', 'y', 'width'])
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


	if conf > 90 and text != " ":
		# print(text, "Y", y, "ΔY", abs(curr_y - y))
		total_width += w

		if abs(curr_y - y) <= line_threshold or curr_y == 0:
			l.append(text)
			x_list.append(x)
		else:
			lines_df = lines_df.append({'text': l.copy(), 'x': min(x_list), 'y': curr_y, 'width': total_width}, ignore_index=True)
			x_list = []
			x_list.append(x)
			# print("Line Finished", l)
			lines.append(l.copy())
			l.clear()
			total_width = 0
			l.append(text)
		curr_y = y

# merge lines into text blocks
curr_y = 0
x = 0
width = 0
text_blocks = []
text_blocks_df = pd.DataFrame(columns=['text', 'x', 'y', 'width'])
string = []

for index, line in lines_df.iterrows():
	y = line['y']
	x = line['x']
	width = line['width']
	if abs(curr_y - y) <= line_break_threshold or curr_y == 0:
		x_list.append(x)
		for word in line['text']:
			string.append(word)
	else:
		text_blocks_df = text_blocks_df.append({'text': ' '.join(string), 'x': min(x_list), 'y': curr_y, 'width': line['width']}, ignore_index=True)
		string = []
		for word in line['text']:
			string.append(word)
		x_list = []
		x_list.append(x)
	curr_y = y

text_blocks_df = text_blocks_df.append({'text': ' '.join(string), 'x': min(x_list), 'y': curr_y, 'width': width}, ignore_index=True)

print(text_blocks_df)