## Short description of the processing pipeline

1. Sample data is pulled from Reddit and classified *manually*
2. Mean image size is found and all images are resized to that size
3. Image size is shrinked to 10% of original size
4. Images are greyscaled and transformed into a list of floats [0-1]
5. Data set is split 80-20 into training and test data
5. SGD Classifier is fitted