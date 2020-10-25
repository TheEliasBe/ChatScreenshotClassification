# Chat Screenshot Classification and Text Localization
This script aims to classify a image into chat screenshot and non chat screenshot. Of the chat screenshot images, the message text is extracted.

## Example

![example text screenshot](https://github.com/TheEliasBe/ChatScreenshotClassification/blob/master/example/0vzd3bcrq8a31.jpg)
After applying the localization script:

```
  x     y  ...                                               text partner
333   690  ...                                               What     0.0
923  1140  ...                         I'll rephrase the question     1.0
571  1388  ...      What do you plan on doing with your time off?     1.0
333  1950  ...                                   Shopping, travel     0.0
571  2647  ...  You're very cute, shame your conversation is l...     1.0
333  3209  ...                                               What     0.0
```

The text in the bubbles is extracted, while additional text (e.g. sent/received acknowledgements, time stamp) is ignored. The text is even assigned to a partner of the conversation.

## Short description of the processing pipeline of the classification

1. Sample data is pulled from Reddit and classified *manually*
2. Mean image size is found and all images are resized to that size
3. Image size is shrinked to 10% of original size
4. Images are greyscaled and transformed into a list of floats [0-1]
5. Data set is split 80-20 into training and test data
5. SGD Classifier is fitted

## Short description of the processing pipeline of text localization

1. Scale the image 2x and convert to grey
2. Apply Tesseract OCR
3. Blur and threshold the image
4. Apply edge detection and find bounding boxes of all separate edges
5. Assign text to bounding boxes
6. Assign bounding boxes to conversation participants
7. Remove stop words
