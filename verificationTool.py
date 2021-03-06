import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.feature_extraction import image
import os.path
import keras
import tensorflow
import sklearn
import sys

# This is the Image Verification Tool class that is used
# to verify whether two images are similar enough. It uses different
# Computer Vision algorithms and methods to perform Feature Extraction, Image Verification
# and Image Segmentation.
#
# @author Preslav Kisyov
# @version 1.1
class ImageVerificationTool:
    # Print library versions
    print("Keras: ", str(keras.__version__), " TF: ", str(tensorflow.__version__), " NumPy: ", str(np.__version__), " OpenCV: ", str(cv2.__version__),
          " Sklearn: ", str(sklearn.__version__), " Argparse: ", str(argparse.__version__),  " Matplotlib: ", str(matplotlib.__version__),
          " Time: ", str(sys.version), " Warnings: ", str(sys.version),  " OS: ", str(sys.version))

    # Ignore CPU Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # This function picks the CNN to be used
    # for feature extraction depending on the argument
    # passed through to the model
    #
    # @param model The model to be used
    def pick_model(self, model):
        # Initialise the model depending on the provided arg
        # Note: Default is always ResNet50, so it cannot be None
        if model == "resnet50":
            from keras.applications.resnet import ResNet50, preprocess_input
            self.resnet = ResNet50(include_top=False, weights="imagenet")
        elif model == "resnet101":
            from keras.applications.resnet import ResNet101, preprocess_input
            self.resnet = ResNet101(include_top=False, weights="imagenet")
        elif model == "resnet152":
            from keras.applications.resnet import ResNet152, preprocess_input
            self.resnet = ResNet152(include_top=False, weights="imagenet")
        elif model == "vgg19":
            from keras.applications.vgg19 import VGG19, preprocess_input
            self.resnet = VGG19(include_top=False, weights="imagenet")
        elif model == "inception":
            from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
            self.resnet = InceptionResNetV2(include_top=False, weights="imagenet")
        self.preprocess_input = preprocess_input

    # Import the query and reference images as an array
    #
    # @param query_path The path to the query image
    # @param refer_path The path to the reference image
    # @return The query image array
    # @return The reference image array
    def read_images(self, query_path, refer_path): return cv2.imread(query_path), cv2.imread(refer_path)

    # Pre-process an image so it can
    # be passed over to a Residual Network.
    # This includes zero-centering the pixels as
    # well as increasing the dimension
    #
    # @param image The image to be pre-processed
    # @return The pre-processed image of shape (1, h, w, 3)
    def process_image(self, image):
        img = np.expand_dims(image, axis=0)
        return self.preprocess_input(img.astype(np.float32))

    # Extract feature vectors from two images using a ResNet
    #
    # @param query_image The query image array
    # @param refer_image The reference image array
    # @return The query feature vector
    # @return The reference feature vector
    def get_features(self, query_image, refer_image):
        query_features = self.resnet.predict(query_image)[0]
        refer_features = self.resnet.predict(refer_image)[0]

        sizes = [[query_features.shape[0], query_features.shape[1]],
                 [refer_features.shape[0], refer_features.shape[1]]]

        query_reshaped = query_features.reshape(query_features.shape[0] *
                        query_features.shape[1], query_features.shape[2])
        refer_reshaped = refer_features.reshape(refer_features.shape[0] *
                        refer_features.shape[1], refer_features.shape[2])
        return query_reshaped, refer_reshaped, sizes

    # Perform Correlation Coefficient on two feature vectors
    # in order to get a similarity vector with best matching features
    #
    # @param query_f The query feature vector
    # @param refer_f The reference feature vector
    # @return cc The Correlation Coefficient vector
    def get_cc(self, query_f, refer_f):
        # Local Center the pixels
        query = query_f - np.mean(query_f, axis=0, keepdims=True)
        refer = refer_f - np.mean(refer_f, axis=0, keepdims=True)

        # Normalize query vector
        query = np.transpose(query)
        query_v = np.sum(np.square(query), axis=0, keepdims=True)

        # Normalize refer vector
        refer_v = np.sum(np.square(refer), axis=1, keepdims=True)

        # Get denominator
        denominator = np.multiply(np.sqrt(refer_v), np.sqrt(query_v))

        # Get Dot Product
        product = np.dot(refer, query)

        # Handle division by 0 or Nan (will return 0)
        np.seterr(divide='ignore', invalid='ignore')

        # Getting Correlation coefficient
        cc = np.divide(product, denominator)
        return cc

    # Perform The Normalized Cross-Correlation on two feature vectors
    # in order to get a similarity vector with best matching features
    #
    # @param query_f The query feature vector
    # @param refer_f The reference feature vector
    # @return ncc The Normalized Cross-Correlation/Cosine Similarity vector
    def get_ncc(self, query_f, refer_f):
        # Normalize query vector
        query = np.transpose(query_f)
        query_v = np.sum(np.square(query), axis=0, keepdims=True)

        # Normalize refer vector
        refer_v = np.sum(np.square(refer_f), axis=1, keepdims=True)

        # Get denominator
        denominator = np.multiply(np.sqrt(refer_v), np.sqrt(query_v))

        # Get Dot Product
        product = np.dot(refer_f, query)

        # Handle division by 0 or Nan (will return 0)
        np.seterr(divide='ignore', invalid='ignore')

        # Getting The Normalized Cross-Correlation/Cosine Similarity
        ncc = np.divide(product, denominator)

        return ncc

    # Extract the likelihood maps for two images
    # from a similarity vector with shape (h*w, h1*w1)
    #
    # @param similarity_v The similarity vector
    # @param sizes The sizes of the query and reference maps
    # @return query_mask The query feature mask
    # @return refer_mask The reference feature mask
    def get_masks(self, similarity_v, sizes):
        # Extract best features from the two axes of the similarity vector
        # Get the biggest values
        query_mask = np.max(similarity_v, axis=0)
        refer_mask = np.max(similarity_v, axis=1)

        # Reshape the masks to be of two dimensions so they can be shown as images
        query_mask = query_mask.reshape(sizes[0][0], sizes[0][1])
        refer_mask = refer_mask.reshape(sizes[1][0], sizes[1][1])
        return query_mask, refer_mask

    # This function extracts the starting and ending
    # points of the query image for it to be presented
    # on the reference image
    #
    # @param loc The location of the object
    # @param w The width of the query image
    # @param h The height of the query image
    # @param scale The scale to have the query mask dimensions reduced
    # @return first_point, second_point The starting and ending points
    def get_points(self, loc, w, h, scale=1):
        first_point = (int(loc[0] - w / scale), int(loc[1] - h / scale))
        second_point = (int(loc[0] + w / scale), int(loc[1] + h / scale))
        return first_point, second_point

    # Show the query and reference masks using matplotlib,
    # as well as their original images
    #
    # @param images The two original images
    # @param masks The two mask vectors
    # @param patch If available, a patch reference image
    def print_masks(self, images, masks, patch):
        q_mask, r_mask = masks

        # If a patch is not available, draw a likelihood object rectangle
        # on the reference image, showing the best matched location
        if patch is None:
            w, h = q_mask.shape[::-1]  # Query image
            _, _, _, loc = cv2.minMaxLoc(r_mask)
            first_point, second_point = self.get_points(loc, w, h)
            cv2.rectangle(r_mask, first_point, second_point, 255, -1)  # Draw the object
        else: r_mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Show the patch image instead

        # Plot the query and reference masks as well as their original images
        fig, axes = plt.figure(figsize=(15, 15)), (2, 2)

        info_map = {0: ["Query original image", images[0]], 1: ["Reference original image", images[1]],
               2: ["Query image mask", q_mask], 3: ["Reference image mask", r_mask]}

        # Creating a subplot for every image
        for index_ax in range(sum(axes)):
            subPlot = fig.add_subplot(axes[0], axes[1], index_ax+1)  # rows; cols; index
            subPlot.title.set_text(info_map[index_ax][0])
            plt.imshow(info_map[index_ax][1], 'gray')
        plt.show()

    # Resize the query image with given certain percentage/scale.
    #
    # @param query The query image to be resized
    # @param scale The percentage to have the image shape rescaled to
    # @return query The newly resized query image
    def get_query_image(self, query, scale):
        width = query.shape[1]*scale/100
        height = query.shape[0]*scale/100
        query = cv2.resize(query, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        return query

    # Perform Patch Matching technique using a query and reference image.
    # This method extracts patches of certain size from the reference image
    # and then extracts features and similarity vector given every patch.
    #
    # @param queryImg The query image array
    # @param referImg The reference image array
    # @return maxSimilarity The best similarity value
    # @return bestCC The best Correlation Coefficient vector
    # @return bestSizes The sizes of the best feature vectors
    # @return patch The best matched patch image
    def patch_matching(self, queryImg, referImg):
        maxSimilarity, bestSizes, bestCC = 0, [], None
        best_patch = None
        # Resize/Pre-process query image
        imgQ = cv2.resize(queryImg, (int(queryImg.shape[1] * float(0.15)), int(queryImg.shape[0] * float(0.15))), interpolation=cv2.INTER_AREA)
        query = self.process_image(imgQ)
        # Get reference patches
        patches = image.extract_patches_2d(referImg, (224, 224), max_patches=250)
        # Iterate over every patch image
        for patch in patches:
            patch_img = self.process_image(patch)
            query_f, refer_f, sizes = self.get_features(query, patch_img)
            # Perform Correlation Coefficient and extract similarity value
            maxSim, cc = self.get_max_value(query_f, refer_f)
            # Update values to the best ones so far
            if maxSim >= maxSimilarity:
                maxSimilarity, bestCC, bestSizes, best_patch = maxSim, cc, sizes, patch

            # Stop performing the method if a match has been found
            if self.check_max_similarity(maxSim): break
        return maxSimilarity, bestCC, bestSizes, best_patch

    # Perform the picked Similarity Measure method
    # and extract the maximum/minimum value from the similarity vector
    #
    # @param query_f The query feature vector
    # @param refer_f The reference feature vector
    # @return A rounded similarity value
    # @return The Similarity Measure vector
    def get_max_value(self, query_f, refer_f):
        # Pick the Similarity Measure to be used given the passed
        # argument. The measure will never be None as it has default value
        measure = None
        if self.measure == "cc": measure = self.get_cc(query_f, refer_f)
        elif self.measure == "ncc": measure = self.get_ncc(query_f, refer_f)
        _, sim, _, _ = cv2.minMaxLoc(measure)

        return round(sim, 3), measure

    # Perform the Template Matching technique variation on a query and reference image.
    # This method rescales the query image to different sizes, using it as a template in order
    # to match it to the reference image.
    #
    # @param queryImg The query image array
    # @param referImg The reference image array
    # @return maxSimilarity The best similarity value
    # @return bestCC The best Correlation Coefficient vector
    # @return bestSizes The sizes of the best feature vectors
    def template_matching(self, queryImg, referImg):
        if sum(queryImg.shape) > sum(referImg.shape):
            scales = [10, 12, 14, 16, 18, 20]
        else: scales = [22, 24, 26, 28, 30]
        maxSimilarity, bestSizes, bestCC = 0, [], None
        imgR = self.process_image(referImg)
        for scale in scales:
            query = self.get_query_image(queryImg, scale)
            query = self.process_image(query)
            query_f, refer_f, sizes = self.get_features(query, imgR)
            # Perform Correlation Coefficient and extract similarity value
            maxSim, cc = self.get_max_value(query_f, refer_f)

            # Update values to the best ones so far
            if maxSim >= maxSimilarity:
                maxSimilarity, bestCC, bestSizes = maxSim, cc, sizes


            # Stop performing the method if a match has been found
            if self.check_max_similarity(maxSim): break

        return maxSimilarity, bestCC, bestSizes

    # Perform a check whether further iteration of either
    # the image scales or patches is required.
    #
    # @param maxSim The current best similarity value
    # @return False If further iterations are required
    # @return True If the iteration process should stop
    def check_max_similarity(self, maxSim):
        if self.extract or self.print_mask or self.isTest: return False  # Do whole iterations when extracting values
        # Predict against threshold
        return True if maxSim >= self.threshold else False

    # The following code is patented, thus it must not be
    # used for commercial use. Similar implementation of the
    # code can also be found at - https://docs.opencv.org/master/d5/d6f/tutorial_feature_flann_matcher.html (OpenCV)
    # The paper of the method is Speeded-Up Robust Features (SURF), with authors - Herbert Bay, Andreas Ess, Tinne Tuytelaars, and Luc Van Gool
    #
    # This function gets the number of matches produced by the method.
    #
    # @params gray_imgQ, gray_imgR The Query and Reference images in gray mode
    # @return The number of matches above a threshold
    def get_surf_value(self, gray_imgQ, gray_imgR):
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400) # Initialise the method
        # Extract features
        _, q_features = surf.detectAndCompute(gray_imgQ, None)
        _, r_features = surf.detectAndCompute(gray_imgR, None)
        feature_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        feature_matches = feature_matcher.knnMatch(q_features, r_features, 2)

        return len([match_pair[0] for match_pair in feature_matches if match_pair[0].distance < 0.75 * match_pair[1].distance])

    # Get the best maximum similarity value by performing different
    # matching methods, depending on the choice
    #
    # @param path_q The path to the query image
    # @param path_r The path to the reference image
    # @param patch Default None, if Patch Matching is performed, it will be an image patch
    # @return best_max_sim the best maximum similarity
    def get_max_similarity(self, path_q, path_r, patch=None):
        imgQ, imgR = self.read_images(path_q, path_r)
        if self.surf:
            gray_imgQ = cv2.cvtColor(imgQ, cv2.COLOR_BGR2GRAY)
            gray_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            # Resize to 80% of the images to avoid memory overflow
            gray_imgQ = cv2.resize(gray_imgQ, (int(gray_imgQ.shape[1] * float(0.80)), int(gray_imgQ.shape[0] * float(0.80))), interpolation=cv2.INTER_AREA)
            gray_imgR = cv2.resize(gray_imgR,
                                  (int(gray_imgR.shape[1] * float(0.80)), int(gray_imgR.shape[0] * float(0.80))),
                                  interpolation=cv2.INTER_AREA)

            # Get the number of matches (the similarity value)
            best_max_sim = self.get_surf_value(gray_imgQ, gray_imgR)
            if self.print_mask: print("Printing has been disabled for SURF because of its different implementation!")
        else:
            # Check matching method
            if self.match_method == "tm": best_max_sim, bestCC, bestSizes = self.template_matching(imgQ, imgR)
            else: best_max_sim, bestCC, bestSizes, patch = self.patch_matching(imgQ, imgR)

            # Extract and draw image masks
            if self.print_mask is True and best_max_sim >= self.threshold:
                images = [cv2.imread(path_q), cv2.imread(path_r)]
                mask_q, mask_r = self.get_masks(bestCC, bestSizes)
                self.print_masks(images, [mask_q, mask_r], patch)

        print("BEST SIMILARITY: ", best_max_sim)
        return best_max_sim

    # Generate a file that can be used to visualise data
    # The file format is as follows:
    # FEATURE LABEL -> Similarity Value ; Prediction
    #
    # @param dataset A list containing the paths to both images
    # as well as the path to the prediction labels
    # @param dataset_path The path for the new generated file
    def create_extr_file(self, dataset, dataset_path):
        self.comp_time = time.time()  # Start counting time
        extr_features, extr_labels = [], []
        pathsQ, pathsR, labels = dataset
        new_data = self.load_dataset(pathsQ, pathsR, labels)

        # Make features and labels lists and print the current progress
        for path_index in range(len(new_data)):
            similarity = self.get_max_similarity(new_data[path_index][0], new_data[path_index][1])
            self.show_progress(path_index+1, len(new_data))
            label = new_data[path_index][2]
            print("Label: "+str(label)+" ...("+str(path_index+1)+"/"+str(len(new_data))+")")
            extr_features.append(similarity)
            extr_labels.append(label)

        # Create an extraction file with values
        try:
            if os.path.exists(dataset_path): print("Extraction file already exists! Overwriting file...!")
            with open(dataset_path, 'w') as f:
                for item in range(len(extr_features)):
                    feature = str(extr_features[item])
                    label = str(extr_labels[item])
                    f.write(feature+" "+label+" \n")
            print("New file created -> "+dataset_path)
        except IOError:


            print("Could not write to " + dataset_path + " file")
            raise
        new_time = self.get_time()
        print("Average computation time: "+str(new_time/len(new_data))+"s")
        print("Test total time: "+str(new_time)+"s")

    # Generate two arrays that contain the features and labels
    # from a specified extraction/test file
    #
    # @param file_path The path to the file
    # @return features A list of features
    # @return labels A list of labels
    def load_from_file(self, file_path):
        features, labels = [], []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    features.append([float(lines[i].split(' ')[0])])
                    labels.append(int(lines[i].split(' ')[1].split('\n')[0]))
            f.close()
        except IOError:
            print("File " + file_path + " not found!")
            raise
        return features, labels

    # Load a file so its content can be plotted on a graph
    #
    # @param file_path The path to the file
    # @return pos_x The positive samples (samples with label 1)
    # @return neg_x The negative samples (samples with label 0)
    def load_for_plot(self, file_path):
        neg_x, pos_x  = [], []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # Depending on the label pick pos/neg samples
                for i in range(len(lines)):
                    if int(lines[i].split(' ')[1].split('\n')[0]) == 0:
                        neg_x.append(float(lines[i].split(' ')[0]))
                    else:
                        pos_x.append(float(lines[i].split(' ')[0]))
            f.close()
        except IOError:
            print("File "+file_path+" not found!")
            raise

        return pos_x, neg_x

    # Plot data given positive and negative samples
    #
    # @param pos_x Positive samples
    # @param neg_x Negative samples
    # @return If no samples could be found
    def plot_data(self, pos_x, neg_x):
        if (len(pos_x) and len(neg_x)) == 0:
            print("Could not find any samples!")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_ylabel(np.arange(0, max(pos_x), 1.0))
        avg_x = np.mean(pos_x)
        avg_y = np.mean(neg_x)

        plt.bar(range(len(pos_x)), pos_x, width=0.4, color="red", label="Similar", edgecolor='red')  # plotting by columns
        plt.bar(range(len(neg_x)), neg_x, width=0.4,  label="Not Similar", color="blue", edgecolor='blue')
        ax.set_title("Threshold Graph", fontweight='bold')
        ax.set_ylabel('Similarity Score', fontweight='bold')
        plt.xlabel('Prediction', fontweight='bold')
        ax.yaxis.grid(True)
        plt.legend(["Similar " + '%.2f'%avg_x+" avrg", "Not Similar " + '%.2f'%avg_y+" avrg"], loc='upper right')
        plt.show()

    # Generate a dataset from the content of a labels file
    # containing QUERY, REFERENCE, LABEL.
    # The result is a list of the full paths for both images as well as the label
    #
    # @param pathQ The path to the query image
    # @param pathR The path to the reference image
    # @param pathL The path to a labels file of format (Q, R, L)
    def load_dataset(self, pathQ, pathR, pathL):
        dataset = []
        try:
            with open(pathL, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_split = line.split(',')
                    if len(line_split) < 3:
                        print("Unsupported labels format!")
                        raise IOError
                    query_image, ref_image, label = line_split[0], line_split[1], int(line_split[2].split("\n")[0])
                    dataset.append([pathQ+query_image, pathR+ref_image, label])
            f.close()
        except IOError:
            print("File not found!")
            raise

        return dataset

    # Load a test dataset and perform testing on it
    #
    # @param test_dataset A list of Q, R, L paths
    # @return The results of testing
    def test_tool(self, test_dataset):
        if not self.check_file([test_dataset[2]]): return
        self.isTest = True
        data = self.load_dataset(test_dataset[0], test_dataset[1], test_dataset[2])
        return self.test(data)

    # This function checks whether a file or list of files
    # exists and will not throw an error.
    #
    # @param files List of files
    # @return If any of the files do not exist - False else True
    def check_file(self, files):
        for file in files:
            if not os.path.isfile(file):
                print("File "+str(file)+" does NOT exist!")
                return False
        return True

    # Perform testing on some data,
    # returning an accuracy value
    #
    # @param data The data to perform testing on ([Q, R, L])
    # @return Accuracy value
    def test(self, data):
        good_predictions, predictions = 0, 1
        neg, pos = [], []
        test_time = 0
        length_data = len(data)
        print("Testing using threshold...")
        for pair in data:
            if not self.check_file([pair[0], pair[1]]): return
            pred, similarity, comp_time = self.predict(pair[0], pair[1])
            print("REF: ", pair[1], " QUERY: ", pair[0])
            print("PRED: ", str(pred), " AND LABEL: ", str(pair[-1]), "\n")

            # Separate predictions to negative and positive
            if pair[-1] == 0: neg.append(similarity)
            else: pos.append(similarity)

            good_predictions += 1 if pred == pair[-1] else 0
            self.show_progress(predictions, len(data))
            print("Current Accuracy: ", good_predictions/float(predictions), '\n')
            predictions, test_time = (predictions + 1), (test_time + comp_time)

        # Uncomment to write samples to file
        # self.write_samples_to_file(neg, pos)

        new_time = test_time / float(length_data)
        print("Average computation time: "+str(new_time)+"s")
        print("Test total time: "+str(test_time)+"s")
        self.isTest = False
        return good_predictions/float(len(data))

    # This function writes the predicted values to two files:
    # negative and positive.
    # This is needed for the thresholdTool script
    # to find the best threshold given these two files.
    #
    # @param neg The list of negative predictions
    # @param pos The list of positive predictions
    def write_samples_to_file(self, neg, pos):
        with open('./negative.txt', 'w') as f:
            for item in neg:
                f.write(str(item)+"\n")
            f.close()
        with open('./positive.txt', 'w') as f:
            for item in pos:
                f.write(str(item)+"\n")
            f.close()

    # Visualize the prediction results
    #
    # @param prediction The prediction result label
    # @param similarity The similarity value for that prediction
    def print_prediction(self, prediction, similarity):
        if prediction == 1:
            print("The query image is present on the reference image with similarity: "+str(similarity)+"!\n")
        else: print("The query image is NOT present on the reference image with similarity: "+str(similarity)+"!\n")

    # This function performs a check on the extension
    # of a file. If the extension is not supported, then
    # the code will exit!
    #
    # @param pathQ The path to the query image
    # @param pathR The path to the reference image
    # @return True if extensions are correct else False
    def check_file_extension(self, pathQ="", pathR=""):
        isExtQ, isExtR = False, False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if ext in pathQ: isExtQ = True
            if ext in pathR: isExtR = True
        if not isExtQ or not isExtR:
            print("Not supported file format!")
            return False
        else: return True

    # Get prediction based on the result of performing
    # Image Verification with different methods
    #
    # @param pathQ The path to a query image
    # @param pathR The path to a reference image
    # @return prediction The prediction label result
    # @return similarity The similarity value for the current prediction
    # @return The time that the prediction took to complete
    def predict(self, pathQ, pathR):
        if not self.check_file_extension(pathQ, pathR): return
        if not self.check_file([pathQ, pathR]): return
        self.comp_time = time.time()  # Start counting time
        similarity = self.get_max_similarity(pathQ, pathR)
        prediction = 1 if similarity >= self.threshold else 0
        self.print_prediction(prediction, similarity)
        return prediction, similarity, self.get_time()

    # Print a progress bar in order
    # to visualize status
    #
    # @param num_of_pred The current number of predictions completed
    # @param total_pred The total number of predictions
    def show_progress(self, num_of_pred, total_pred):
        current_status = int(round(50 * num_of_pred / float(total_pred)))
        percentage = round(100.0 * num_of_pred / float(total_pred), 1)
        print("[" + '=' * current_status+ "-" * (50 - current_status) + "] "+str(percentage) + "%", end="\n", flush=True)

    # Get the finish time of a process
    #
    # @return new_time The finish time of the process
    def get_time(self):
        new_time = int(time.time() - self.comp_time)
        print("=======================")
        print("Predict Time: "+str(new_time)+"s")
        return new_time

    # This is the constructor for the class
    # It initializes all class variables
    #
    # @param extr_path The path to an extraction file
    # @param extract A boolean value specifying whether to extract values or not
    # @param extr_dataset A list of [Q R L] destinations
    # @param plot A boolean value specifying whether to plot data or not
    # @param threshold A threshold value
    # @param print_mask A boolean value specifying whether to show a likelihood map or not
    # @param match_method A string specifying the matching method (tm or pm)
    # @param model The CNN used to extract features
    # @param measure The Similarity Measure that will be used
    def __init__(self, extr_path, extract, extr_dataset, plot,
                 threshold, print_mask, match_method, surf, model, measure):
        self.isTest = False
        self.measure, model = measure[0], model[0]
        self.pick_model(model)
        self.threshold, self.predict_method, self.match_method = threshold, "threshold", match_method
        self.extract, self.knn, self.print_mask, self.surf = extract, None, print_mask, surf
        print("Initializing tool with: threshold = " + str(threshold) + " | predict_method = " +
              self.predict_method + " | match_method = " + match_method + " | SURF = " + str(surf) +
            " | model = " + str(model) + " | measure = " + str(self.measure))

        # Try to generate an extraction file
        if extract:
            print("Trying to create a new extraction file...")
            if len(extr_dataset) == 0: print("Please provide a valid extraction dataset!")
            else:
                print("New extraction file -> " + extr_path + "!")
                print("Extracting...")
                self.create_extr_file(extr_dataset, extr_path)
        # Try to plot data
        if plot:
            pos_x, neg_x = self.load_for_plot(extr_path)
            self.plot_data(pos_x, neg_x)

# The main method of the python file
# It gets the received arguments from the parser
# and creates an Image Verification Tool object
if __name__ == '__main__':
    # Convert string argument to boolean
    #
    # @param arg The argument from the parser
    # @return True/False or raise exception depending on argument
    def str_converter(arg):
        ex, arg_l = argparse.ArgumentTypeError('Boolean value True or False expected, ('+str(arg)+') given!'), arg.lower()
        if type(arg) is bool: return arg
        if arg_l not in ['true', 'false']: raise ex
        else: return True if arg_l == 'true' else False

    # Initialize Parser Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ep', '--extr_path', action='store', dest='extr_path', help="Specify extraction file path", default="new_extracted_dataset.txt")
    parser.add_argument('-e', '--extract', help="Toggle extraction mode True or False. For False just omit command.", type=str_converter, nargs='?', default=False, const=True)
    parser.add_argument('-ed', '--extr_dataset', dest='extr_dataset', help="Specify an extraction dataset with 3 arguements:\
                                                                                              \n -QUERY_IMAGES DESTINATION FOLDER\
                                                                                              \n -REFERENCE_IMAGES DESTINATION FOLDER\
                                                                                              \n -LABEL FILE DESTINATION", default=[], nargs=3)
    parser.add_argument('--plot', dest='plot', help="Toggle the plot option True or False. Omit command for False", type=str_converter, nargs='?', default=False, const=True)
    parser.add_argument('-thr', '--threshold', action='store', dest='threshold', help="Give a threshold value i.e. 0.50", type=float, default=0.585)
    parser.add_argument('-mm', '--match_method', action='store', dest="match_method", help="Specify a matching method\
                                                                                                  to be used from [\
                                                                                                  'tm' -> for template matching, 'pm -> patch matching']. If \
                                                                                                  None, default is template matching (tm)\
                                                                                                  method", type=str,
                                                                                                                default="tm")
    parser.add_argument('-pm', '--print_mask', help="Toggle print mask option True or False. For False just omit command.\
                                                    \nThis option prints only images that are similar and only when predicting!",
                                                    type=str_converter, nargs='?', default=False, const=True)
    parser.add_argument('-test','--test_dataset', help="Test on a specified dataset of 3 arguements:\
                                            \n -QUERY_IMAGES DESTINATION FOLDER\
                                            \n -REFERENCE_IMAGES DESTINATION FOLDER\
                                            \n -LABEL FILE DESTINATION", default=[], nargs=3)
    parser.add_argument('-p', '--predict', help="Predict on a new pair of Query and Reference Images. \
                                                     Specify the path of a QUERY and a REFERENCE image.", default=[],
                        nargs=2)
    parser.add_argument('-s', '--surf', help="Toggle surf comparison mode True or False. For False just omit command.",
                        type=str_converter, nargs='?', default=False, const=True)
    parser.add_argument('--model', choices=['resnet50', 'resnet101', 'resnet152', 'vgg19', 'inception'], nargs=1, default=['resnet50'])
    parser.add_argument('--measure', choices=['cc', 'ncc'], nargs=1, default=['cc'])
    # Get a dictionary of parser arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Remove the exceptions
    filtered_dict = dict((k, args_dict[k]) for k in args_dict.keys() if k not in ["test_dataset", "predict"])

    # Create class instance with specified arguments
    class_instance = ImageVerificationTool(**filtered_dict)

    # Perform class functions given specified arguments
    if args.predict:
        print("Predicting...")
        class_instance.predict(args.predict[0], args.predict[1])
    if args.test_dataset:
        class_instance.test_tool(args.test_dataset)
