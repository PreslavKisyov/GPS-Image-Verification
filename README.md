# Image-Based GPS Verification README #
This project utilises different AI and Computer Vision methods
to solve the long-pressing issue of Image Verification. The approach
taken is to compare a query and a reference image taken from some coordinates
and extract a similarity value. Additionally, verification is performed by comparing
that value against a threshold.
- The best threshold is picked by running the thresholdTool.py
- The image verification is performed by running the verificationTool.py

# Installation #
Python 3 is required in conjunction with 'pip'.
To install all libraries run - pip install -r requirements.txt
This will get all the requirements needed and install them recursively.
Note: For some systems, administrator privilages might be needed.

