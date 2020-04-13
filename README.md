# Image-Based GPS Verification README #
This project utilises different AI and Computer Vision methods
to solve the long-pressing issue of Image Verification. The approach
taken is to compare a query and a reference image taken from some coordinates
and extract a similarity value. Additionally, verification is performed by comparing
that value against a threshold.<br/>
- The best threshold is picked by running the **thresholdTool.py**<br/>
- The image verification is performed by running the **verificationTool.py**<br/>

# Installation #
**Python 3** is required for this project. The version used in this project is 3.6.8.
**Python 3** can be downloaded from **https://www.python.org/downloads/**. 
There are two ways to install the project:
1) Run the **setup.py** file present in the main directory of the project.
The command is - **python setup.py install**.
**Note:** If python 2 is also present on the machine, the command should
look like - **python3 setup.py install**. And that would be valid for all the commands in this project.
2) If the previous method did not work, there is an alternative.
It uses '**pip**', and more specifically **pip3**, as python 3 is used.<br/>
The guide to installing **pip** can be found on the following website - **https://pip.pypa.io/en/stable/installing/**.
Moreover, the **pip** file is provided in the main directory of the project. It is called **get-pip.py**, and it has been provided by the url mentioned above.
The command to install that file is as follows:<br/>
- **python get-pip.py**<br/>
For more installation options, the provided guide above can be referenced.<br/>
To install all libraries run - **pip install -r requirements.txt**.<br/>
This will get all the requirements needed and install them recursively.
Additionally, the requirements can be installed manually. <br/>
An issue could occur on some systems after installing the Tensorflow library, especially if it was already present on the machine. <br/>
The error would say that the gast library is missing, although it could be installed. That is because different versions are not compatible together. <br/>
The fix is rather simple. The gast library must be removed and then installed again:<br/>
- **pip uninstall gast**<br/>
- **pip install gast**<br/>
:warning:**Note:** For some systems, administrator privileges might be required.<br/>

# verificationTool User Guide #
The model implements multiple arguments and commands that can be invoked by running them in the terminal. The commands shown here are for the test data provided in the ‘Test’ folder. However, these commands can be used on custom data as well. Additionally, there could be multiple configurations of the available parameters and they can be used in conjunction with other parameters. All of the following commands have been run from the current directory.

## Predict ##
:exclamation:**Template Matching Prediction** – python verificationTool.py -p ./Test/images/image1Q.jpg ./Test/images/image1R.jpg -mm tm<br/>
:exclamation:**Patch Matching Prediction** - python verificationTool.py -p ./Test/images/image2Q.jpg ./Test/images/image2R.jpg -mm pm<br/>
:warning:To print the likelihood maps, add the **-pm** command to either of the commands above. A figure should appear on the screen if and only if the prediction is positive, otherwise it would not print anything!<br/>
:warning:To change the threshold used, add the **-thr someNumber** command to either of the commands above!

## Test ##
:exclamation:**Template Matching Test** - python verificationTool.py -test “path to query images” “path to reference images” “path to labels.txt file” -mm tm<br/>
:exclamation:**Patch Matching Test** - python verificationTool.py -test “path to query images” “path to reference images” “path to labels.txt file” -mm pm<br/>
:warning:Again, the user can change the threshold by adding the **-thr someNumber** command to either of the commands above!

## Extract Similarity Values to File and Plot ##
:exclamation:**Extract Values** - python verificationTool.py -e -ed "path to query images" "path to reference images" "path to labels" -ep "path to the new text file”<br/>
The **-e** command invokes the extraction mode, **-ed** command provides the query and reference images, as well as the labels text file, and **-ep** provides the path where the new file will be saved to.<br/>
:exclamation:**Plot Values** - python verificationTool.py --plot -ep "path to a text file with similarity values"<br/>
The **--plot** command is used with the **-ep** command. Here the **-ep** command provides the text file and it is not used to create a new one. The result of plotting is shown in the paper of this project and should produce a graph of the values.

## SURF ##
To use the SURF method, simply invoke the **-s** command after either any of the prediction or testing commands. It would invoke the SURF method that is also described and referenced in the paper of this project.<br/>
:warning:**Note:** It does not use any of the methods invoked from the -mm command. Thus, it must be omitted.<br/>
:exclamation: python verificationTool.py -test “path to query images” “path to reference images” “path to labels.txt file” -s

### Additional Information ###
The labels text file when extracting or plotting should have the following layout:<br/>
**SimilarityValue Label**<br/>

**0.50 1**<br/>
**0.30 0**<br/>
The labels text file when testing should have the following layout:<br/>
**Query,Reference,Label**<br/>
**queryImage.jpg,referenceImage.jpg,0**<br/>
:warning:**Note:** The full paths are received by the other two arguments when invoking the -test command (query images path, reference images path).

# thresholdTool User Guide #
This is the script that iterates over multiple thresholds and picks the best one for the data provided. It requires two files. One with negative sample similarity values and one with positive ones. Additionally, the **-r** command can be invoked to reverse the order of importance. Meaning, lower values will be better. This mode can be invoked depending on the similarity measure used for extracting the similarity values.<br/>
:exclamation:**Get Best Threshold Default** – python thresholdTool.py -a “path to negative text file” “path to positive text file”<br/>
:warning:An extra argument can be added that specifies the location of where the produced plot graph will be saved. If not specified, it will be saved to a **default.jpg** image!<br/>
:exclamation:**Get Best Threshold Reverse Order** - python thresholdTool.py -a “path to negative text file” “path to positive text file” “path to a new jpg file” -r

### Additional Information ###
Both files must be passed through in that exact order. The negative file comes before the positive one, as shown in the examples above. Both files should have the following layout:<br/>
**SimilarityValue**<br/>

**0.50**<br/>
**0.60**<br/>
To extract those values, there is a commented-out method in the **verificationTool.py**, that saves two files given the values predicted by the model. It can be found in the **test** function.

# List of Used Libraries #
**Keras** - 2.3.1<br/>
**Tensorflow** - 2.0.0<br/>
**NumPy** - 1.16.0<br/>
**OpenCV** - 3.4.2.16<br/>
**Sklearn** - 0.22.1<br/>
**Argparse** - 1.1<br/>
**Matplotlib** - 3.1.0<br/>
**Time** - built into python's interpreter (python 3.6.8)<br/>
**Warnings** - built into python's interpreter (python 3.6.8)<br/>
**OS** - built into python's interpreter (python 3.6.8)
