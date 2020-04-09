##### THIS IS THE README FILE FOR THE THRESHOLD FOLDER #####

This folder has 4 separate text files that contain the values obtained from running the best configurations
on both Patch Matching and Template Matching methods. Each method has a negative and a positve file, which contains the
corresponding negative or positive sample predictions. These files are meant to be run by the thresholdTool.py script, and should
produce the same graphs as in the paper of the project. The files are:

negbestpm.txt - The negative samples for the Patch Matching method
posbestpm.txt - The positive samples for the Patch Matching method

negbesttm.txt - The negative samples for the Template Matching method
posbesttm.txt - The positive samples for the Template Matching method

Note: The full paths of these files must be passed through to the thresholdTool.py script. Additionally, 
the negative file must be passed first, before the positive one.

############################################################