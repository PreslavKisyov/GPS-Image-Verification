<p align="center">
  <img src=https://github.com/PreslavKisyov/GPS-Image-Verification/blob/master/Capture.JPG />
</p>

# Overview
This project utilises different AI and Computer Vision methods to try and solve the long-pressing issue of Image Verification. The approach taken compares a query with a reference image taken from some coordinates, outputting a `similarity value`. Additionally, verification is performed by comparing that value against a predefined threshold.
- The best threshold is picked by running the `thresholdTool.py`
- The image verification is performed by running the `verificationTool.py`

# Requirements
- **Python 3** - 3.6.8 or higher
- **Pip**

# Installation #
To install the project, the following steps can be followed:
- Creating a virtual environment
  - **Python 3 virtual environment** - creating a virtual environment can help preserve the system from installing and uninstalling different requirements, as well as it will provide a clean installation process. To create that environment, the following command can be executed on a terminal - **python3 -m venv verification-env**.
The name of the environment is specified after the `venv` command. In this case it is `verification-env`. The environment will create a directory at the current location that contains a copy of the python interpreter. After successfully creating the environment, it needs to be activated. This can be done with the following commands:<br/>
    - **Windows** - verification-env\Scripts\activate.bat
    - **Unix or MacOS** - source verification-env/bin/activate</br>
- Installing requirements
  - The requirements can be installed with the help of pip, or more specifically pip3 - **pip install -r requirements.txt**.
- An issue could occur on some systems after installing the Tensorflow library, especially if it was already present on the machine. The error would say that the `gast` library is missing, although it could be installed. That is because different versions are not compatible together. The gast library must be removed and then installed again:<br/>
  - **pip uninstall gast**<br/>
  - **pip install gast**
  
**Note:** For some systems, administrator privileges might be required. Additionally, the issues described above should not occur if installed in a virtual environment<br/>

# verificationTool User Guide #
The model implements multiple arguments and commands that can be invoked by running them in the terminal. The commands shown here are for the test data provided in the `Test` folder. However, these commands can be used on custom data as well. Additionally, there could be multiple configurations of the available parameters and they can be used in conjunction with other parameters. All of the following commands have been run from the current directory.

## Predict ##
**Template Match Prediction** – `python verificationTool.py -p ./Test/images/image1Q.jpg ./Test/images/image1R.jpg -mm tm`<br/>
**Patch Match Prediction** - `python verificationTool.py -p ./Test/images/image2Q.jpg ./Test/images/image2R.jpg -mm pm`

Another argument can be added to the command above that specifies either the model used **--model** or the similarity measure **--measure**.<br/>
For the first command the choices are - ['resnet50', 'resnet101', 'resnet152', 'vgg19', 'inception']. The default is resnet50.<br/>
An example would be - python verificationTool.py -p ./Test/images/image1Q.jpg ./Test/images/image1R.jpg -mm tm --model resnet152<br/>
The second command (**--measure**) has 2 choices - ['cc', 'ncc']. These choices represent Correlation Coefficient and Normalized Cross-Correlation respectively.<br/>
An example of that command would be - python verificationTool.py -p ./Test/images/image1Q.jpg ./Test/images/image1R.jpg -mm pm --model vgg19 --measure ncc<br/>
:warning:To print the likelihood maps, add the **-pm** command to either of the commands above. A figure should appear on the screen if and only if the prediction is positive, otherwise it would not print anything!<br/>
:warning:To change the threshold used, add the **-thr someNumber** command to either of the commands above!

## Test ##
:exclamation:**Template Matching Test** - python verificationTool.py -test “path to query images” “path to reference images” “path to labels.txt file” -mm tm<br/>
:exclamation:**Patch Matching Test** - python verificationTool.py -test “path to query images” “path to reference images” “path to labels.txt file” -mm pm<br/>
:warning:Again, the user can change the threshold by adding the **-thr someNumber** command to either of the commands above, as well as change the model or similarity measure used!

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
:exclamation:**Get Best Threshold Reverse Order** - python thresholdTool.py -a “path to negative text file” “path to positive text file” “path to a new jpg file” -r <br/>
:exclamation:**Adjust Threshold Intensity Value** - python thresholdToold.py -a "path to negative sample values file" "path to positive sample values file" -ints 0.0001 
### Additional Information ###
Both files must be passed through in that exact order. The negative file comes before the positive one, as shown in the examples above. Both files should have the following layout:<br/>
**SimilarityValue**<br/>

**0.50**<br/>
**0.60**<br/>
To extract those values, there is a commented-out method in the **verificationTool.py**, that saves two files given the values predicted by the model. 
It can be found in the **test** function. 
Additionally, the path files of the two files can be changed in the acutal function:<br/>
# Uncomment to write samples to file<br/>
# self.write_samples_to_file(neg, pos)<br/>
Here the seconf line must be uncommented in order to enable the function. In the following code, the path of the two files
can be changed: <br/>
def write_samples_to_file(self, neg, pos):<br/>
     with open('./negative.txt', 'w') as f:<br/>
         for item in neg:<br/>
             f.write(str(item)+"\n")<br/>
         f.close()<br/>
     with open('./positive.txt', 'w') as f:<br/>
         for item in pos:<br/>
             f.write(str(item)+"\n")<br/>
         f.close()<br/>
#Testing Datasets – Download & Evaluation#
To download the **Caltech Buildings dataset**, the following URL must be followed - **http://www.mohamedaly.info/datasets/caltech-buildings**<br/>
There, a download link could be found. The dataset is 195MB. A labels text file will be provided in the **Datasets** folder of the project. The file is called **labels.txt**. <br/>
To test on that dataset (for the best configuration) the following command can be executed:<br/>
**python verificationTool.py --test ./caltech-buildings/ ./caltech-buildings/ ./Datasets/Caltech/labels.txt -mm tm -thr 0.679**<br/>
To download the **Wiki_Commons** dataset is trickier. 
The authors of the **BUPM** paper has provided the required files as well as the download guide for their dataset. <br/>
It can be found at this URL **https://gitlab.vista.isi.edu/chengjia/image-GPS**<br/>
Some of the images might be corrupted or removed from the database, hence they need to be removed from the respective folders. 
The query images must be separated from the reference images as they have the same names. <br/>
They must be put in two separate folders. Because the **Google API** is a paid service, the following tool can be used to download the reference images - **https://svd360.istreetview.com/**
Additionally, the labels.txt file can be found under the WikiCommons in the **Datasets** folder. 
To run the best recorded configuration, the following command can be executed:<br/>
python verificationTool.py --test ./wiki_commons/queries/ ./wiki_commons/references/ ./Datasets/WikiCommons/labels.txt -mm pm -thr 0.652<br/>
The commands that execute testing for both datasets can be altered to match the actual file paths, as they are only given as an example. 
If the datasets are downloaded not in the main directory of the project, then the paths to them must be changed accordingly. 
Additionally, there is an alternative method of downloading both datasets. 
The following **URL** can be followed to download both datasets used immediately:<br/>
**https://emckclacmy.sharepoint.com/:f:/g/personal/k1763856_kcl_ac_uk/EiSS6CNIVuRFudp28yFeRfwBUgyjEnLCA_8E nWeGMwo94g?e=ctSrOB** <br/>
It contains two folders called "caltech-buildings" and "wiki_commons" respectively. 
They contain the images used for testing this model. The whole folders must be downloaded, so the commands described above can be executed without errors. 
The naming of the folders is already set to the one used for the commands. If the former methods are chosen then it 
must be noted that not all images from the **Wiki_Commons** dataset have been used, as also stated in **Chapter 4 of the paper**. 
Additionally, some of the images might not be able to download. 
Therefore, the former method is preferred, as all the images are provided at a single One Drive location. 

##Additional Information##
The paths files provided in both commands can vary depending on where the datasets have been downloaded and how have they been named. Moreover, the paper can be followed in order to change the configurations to match the runs tested in the paper. Every figure in Chapter 4 (Evaluation) provides information about the configuration.
Arguments Limitation in Testing<br/>
There are certain limitations to the arguments that the model accepts. In order to fully replicate the test runs provided in the paper, some of the code must be changed. For instance, if the configuration is about the Patch Matching method, the following lines of code can be modified:
**imgQ = cv2.resize(queryImg, (int(queryImg.shape[1] * float(0.15)), int(queryImg.shape[0] * float(0.15))), interpolation=cv2.INTER_AREA)**<br/>
**patches = image.extract_patches_2d(referImg, (224, 224), max_patches=250)**<br/>
All lines can be found under the patch_matching function. The first line defines the size by which the query image is resized. 
For instance, if the configuration says that the query image has been resized by 20% then the float 0.15 must be changed to 0.20. The third line defines the number of patches used (max_patches). If the configuration uses 150 patches then the 250 must be changed respectively.
If the Template Matching is being used, then several other things must be changed. The lines that must be change can be found under the template_matching function:<br/>
**if sum(queryImg.shape) > sum(referImg.shape):**<br/>
    **scales = [10, 12, 14, 16, 18, 20]**<br/>
**else: scales = [22, 24, 26, 28, 30]**<br/>
For instance, if the configuration uses only one scale range then all lines must be removed and replaced with only:<br/>
**scales = [13,14,15,16,17]** // Depending on the configuration selected<br/>

# List of Used Libraries #
**Keras** - 2.3.1<br/>
**Tensorflow** - 2.0.1<br/>
**NumPy** - 1.16.0<br/>
**OpenCV** - 3.4.2.16<br/>
**Sklearn** - 0.22.1<br/>
**Argparse** - 1.1<br/>
**Matplotlib** - 3.1.0<br/>
**Time** - built into python's interpreter (python 3.6.8)<br/>
**Warnings** - built into python's interpreter (python 3.6.8)<br/>
**OS** - built into python's interpreter (python 3.6.8)

#
**All the methods used in this README, as well as all the papers referenced, have been given credit to and have been cited accordingly in the original paper/report of this project!**
