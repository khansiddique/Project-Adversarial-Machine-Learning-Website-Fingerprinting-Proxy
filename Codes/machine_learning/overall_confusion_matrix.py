from scapy.all import *
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import support_functions as SUP
from numpy import array
import write_file as WF
import log_dataset_csv as csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
# from sklearn.metrics import plot_confusion_matrix
import statistics
import os
from itertools import product
from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier

def read_matrixes(algo_name):
	curr_dir = os.getcwd()
	sourcepath = os.listdir(os.path.join(curr_dir, "matrixes"))
	nonnormalized_file_names = []
	for sfile in sourcepath:
		split_file_name = re.split("_", sfile)
		if "nonnormalized.csv" in split_file_name and algo_name in split_file_name:
			nonnormalized_file_names.append(sfile)
	return nonnormalized_file_names

if __name__ == "__main__":

	# df_y = pd.read_csv('y_data.csv')
	# classes = df_y.to_numpy()
	# red_classes = np.unique(classes)
	# print(len(red_classes))
	# print(red_classes)
	# s = ""
	# for t in red_classes:
	# 	s += "'" + t + "', "
	# print(s)
	algo_list = ["knn", "forest", "svm", "neural"]
	for i in range(4):
		algo_name = algo_list[i]

		nonnormalized_file_names = read_matrixes(algo_name)
		curr_dir = os.getcwd()
		
		df_nonnormalized = pd.read_csv(curr_dir + "/matrixes/" + nonnormalized_file_names[0],header=None)
		s = (df_nonnormalized.shape[0], df_nonnormalized.shape[1])
		all_matrix = np.zeros(s)
		for file_name in nonnormalized_file_names:
			df_nonnormalized = pd.read_csv(curr_dir + "/matrixes/" + file_name,header=None)
			temp_matrix = df_nonnormalized.to_numpy()
			#print(temp_matrix)
			print(file_name)
			all_matrix = np.add(all_matrix, temp_matrix)
		# print(all_matrix.astype(int))

		class_names = ["https://www.google.com/","https://www.facebook.com/","https://www.heise.de/","https://www.b-tu.de/","https://www.amazon.com/","https://www.dotdash.com/","https://www.ebay.com/","http://www.spiegel.de/","https://www.bahn.com/i/view/index.shtml","https://en.wikipedia.org/wiki/Main_Page","https://twitter.com/","https://www.instagram.com/","https://www.linkedin.com/","https://www.flickr.com/","https://www.microsoft.com/de-de/","https://www.reddit.com/","https://www.nytimes.com/","https://www.jimdo.com/","https://www.blogger.com/","https://github.com/","https://www.mozilla.org/","https://myspace.com/","https://www.theguardian.com/","https://edition.cnn.com/us","https://www.paypal.com/","https://www.huffingtonpost.de/","https://www.dropbox.com/","https://www.xing.com/","https://www.owasp.org/index.php/Main_Page","https://www.skype.com/","https://www.springer.com/","https://www.android.com/","https://www.office.com/","https://www.alexa.com/","https://www.netflix.com/","https://opensource.org/","https://www.oracle.com/sun/index.html","https://www.snapchat.com/","https://www.symantec.com/","https://www.avira.com/","https://us.norton.com/","http://www.aboutads.info/","https://www.python.org/","https://siteorigin.com/","http://ewebdevelopment.com/","https://www.justice.gov/","https://www.warnerbros.com/","https://java.com/","https://www.playstation.com/","https://www.iso.org/home.html","https://moodle.org/","https://web.de/","https://www.gmx.net/","https://www.msn.com/","https://www.unicef.org/","https://www.nist.gov/","https://www.dmca.com/","http://taz.de/","http://www.faz.net/aktuell/","https://www.whatsapp.com/","https://www.samsung.com/us/","http://www.kicker.de/","https://translate.google.de/","https://www.transfermarkt.de/","https://www.sparkasse.de/","https://www.deutsche-bank.de/","https://www.ing-diba.de/","https://www.dkb.de/","https://www.commerzbank.de/","https://www.dhl.de/","https://www.wolframalpha.com/","https://www.history.de/","https://www.sueddeutsche.de/","https://www.bild.de/","https://www.chefkoch.de/","http://derbiokoch.de/","https://www.tagesschau.de/","https://www.wetteronline.de/","https://www.wetter.com/","https://prezi.com/","https://www.mindmeister.com/","https://www.notebooksbilliger.de/","https://de.napster.com/","https://www.deezer.com/de/","https://www.spotify.com/","https://soundcloud.com/","https://www.deutschlandradio.de/","http://www.surfmusik.de/","https://www.kino.de/","https://www.cinema.de/","https://www.imdb.com/","http://www.netzkino.de/","https://www.watchbox.de/","https://www.expedia.de/","https://www.opodo.de/"]
		percentages = []
		all_matrix_normalized = np.zeros(s)
		for index1, y in enumerate(all_matrix):
			line_sum = 0
			for index2, x in enumerate(y):
				line_sum += x
			if line_sum == 0:
				line_sum = 1
			all_matrix_normalized[index1][index1] = all_matrix[index1][index1] / line_sum
			percentages.append(all_matrix_normalized[index1][index1])
		# print(all_matrix_normalized)


		pd.DataFrame(all_matrix.astype(int)).to_csv(curr_dir + "/overall_matrix.csv", header=None, index=None)
		pd.DataFrame(all_matrix_normalized).to_csv(curr_dir + "/overall_matrix_normalized.csv", header=None, index=None)

		# print(len(class_names))
		# print(percentages)
		index = np.arange(len(class_names))
		plt.bar(index, percentages)
		plt.xlabel('Websites', fontsize=15)
		plt.ylabel('Percentages', fontsize=15)
		plt.xticks(index, class_names, fontsize=8, rotation=90)
		plt.title('Websites accuracy by: ' + algo_name, fontsize=15)
		plt.show()

	 # df_X = pd.read_csv(file_name_conf_norm)
	    # df_y = pd.read_csv(file_name_conf_not_norm)




  
