import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# This is the class that picks the
# best accuracy and threshold for specifically
# provided data from two files, containing respectively
# positive and negative similarity value samples
#
# @author Preslav Kisyov
# @version 1.1
class BestThresholdTool:

    # This function appends the lines from a file
    # to a specified list
    #
    # @param lines The lines obtained from a file
    # @param l The list that will hold the passed lines
    def append_to_list(self, lines, l):
        for line in lines:
            n_line = line.split("\n")[0]
            l.append(float(n_line))

    # This function gets the samples from two files
    # into two separate lists for both positive
    # and negative samples
    #
    # @param neg_path The path to the file with negative samples
    # @param pos_path The path to the file with positive samples
    # @return neg The newly created list with negative samples
    # @return pos The newly created list with positive samples
    def get_samples(self, neg_path, pos_path):
        neg, pos = [], []
        for path in [neg_path, pos_path]:
            with open(path, "r") as f:
                lines = f.readlines()
                if len(neg) > 0: self.append_to_list(lines, pos)
                else: self.append_to_list(lines, neg)
            f.close()
        return neg, pos

    # This function loops through different thresholds
    # over two sample lists in order to find the best
    # accuracy and threshold for the data
    #
    # @return best_thr The best threshold recorded
    # @return acc The best accuracy recorded
    # @return neg The list with negative samples only
    # @return pos The list with positive samples only
    def get_best_thr(self):
        neg, pos = self.get_samples(self.neg_path, self.pos_path)
        best_thr, acc = 0, 0

        # Get the best threshold/accuracy by trying
        if self.reverse:
            # Get the threshold for measures that where lower values are better
            for thr in np.arange(0.0, 1.0, 0.000001):
                acc_samples = 0
                for n in neg:
                    if n >= thr: acc_samples += 1
                for p in pos:
                    if p < thr: acc_samples += 1
                if acc_samples >= acc: best_thr, acc = thr, acc_samples
        else:
            # Get the threshold for measures that where higher values are better
            for thr in np.arange(0.0, 1.0, 0.000001):
                acc_samples = 0
                for n in neg:
                    if n < thr: acc_samples += 1
                for p in pos:
                    if p >= thr: acc_samples += 1
                if acc_samples >= acc: best_thr, acc = thr, acc_samples

        return best_thr, acc, neg, pos

    # This function plots the data of two
    # sample lists as well as the recorded
    # accuracy and best threshold
    def plot(self):
        print("Finding best threshold...")
        best_thr, acc, neg, pos = self.get_best_thr()
        print("Plotting...")
        sum_len = len(neg) + len(pos)
        x1 = np.linspace(0, sum_len, len(neg))
        x2 = np.linspace(0, sum_len, len(pos))
        plt.plot(x1, neg)
        plt.plot(x2, pos)
        print("Best accuracy: ", acc / sum_len, " Best Threshold: ", best_thr)
        fontP = FontProperties()
        fontP.set_size('small')
        x_coordinates = [0, sum_len]
        y_coordinates = [best_thr, best_thr]
        print("Saving plot to "+self.save_file)
        plt.plot(x_coordinates, y_coordinates)
        plt.legend(['Negative Samples', 'Positive Samples'], loc='upper left')
        plt.title("Acc: " + str(float(round(acc / sum_len, 3))) + " Thr: " + str(float(round(best_thr, 3))))
        plt.xlabel("Samples")
        plt.ylabel("Similarity Values")
        plt.savefig(self.save_file)
        plt.show()

    # The constructor of the class
    # It manages arguments as well as
    # calls other class functions
    #
    # @parameter dataset A list of arguments
    def __init__(self, dataset, reverse):
        self.reverse = reverse
        if len(dataset) == 0: exit()
        elif len(dataset) == 2:
            self.neg_path, self.pos_path = dataset
            self.save_file = "default.jpg"
        else: self.neg_path, self.pos_path, self.save_file = dataset
        print("Initialising class with: | REVERSE: " + str(self.reverse) + " | NEGATIVE FILE: " + self.neg_path + " \n| POSITIVE FILE: " + self.pos_path + " | SAVEFILE LOCATION: " + self.save_file)
        if '.txt' not in self.neg_path:
            self.neg_path = self.neg_path + ".txt"
        if '.txt' not in self.pos_path:
            self.pos_path = self.pos_path + ".txt"
        if ('.jpg' or '.png' or 'jpeg') not in self.save_file:
            self.save_file = self.save_file+".jpg"

        self.plot()

# This is the main method of the python file
# that gets the received arguments from a
# parser and creates an object of the Best
if __name__ == '__main__':
    # Convert string argument to boolean
    # This is the same method used in the verificationTool.py
    #
    # @param arg The argument from the parser
    # @return True/False or raise exception depending on argument
    def str_converter(arg):
        ex, arg_l = argparse.ArgumentTypeError(
            'Boolean value True or False expected, (' + str(arg) + ') given!'), arg.lower()
        if type(arg) is bool: return arg
        if arg_l not in ['true', 'false']:
            raise ex
        else:
            return True if arg_l == 'true' else False
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', '--argument', dest='dataset', help="Specify the file locations in a dataset with 3 arguements:\
                                                                                        \n -NEGATIVE SAMPLES DESTINATION FILE\
                                                                                        \n -POSITIVE SAMPLES DESTINATION FILE\
                                                                                        \n -NEW IMAGE SAVE FILE DESTINATION", default=[], nargs='*')
    parser.add_argument('-r', '--reverse', help="Toggle reverse mode. This should be used when trying to get the best threshold where the lower a value is, the better.",
                        type=str_converter, nargs='?', default=False, const=True)
    # Get a dictionary of parser arguments
    args = parser.parse_args()
    args_dict = vars(args)
    obj = BestThresholdTool(**args_dict)