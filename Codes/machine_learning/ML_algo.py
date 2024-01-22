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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.array(a)[p.astype(int)], np.array(b)[p.astype(int)]

def neural_network(X, y, algo_name):
	#clf = MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(500, 500,500), random_state=0)
	
	clf = MLPClassifier(activation='logistic', solver='sgd', alpha=0.0001,hidden_layer_sizes=(500, 20), batch_size='auto',
	 beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=False, tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

	run_algo(clf, X, y, algo_name)

def random_forest(X, y, algo_name):
	clf = RandomForestClassifier(max_depth=100, random_state=0)
	run_algo(clf, X, y, algo_name)

def cross_val(clf, X, y):
	scores1 = cross_val_score(clf, X, y, cv=10)
	print(scores1)
	print("Accuracy of K-fold: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

	# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	# scores2 = cross_val_score(clf, X, y, cv=cv)
	# print(scores2)
	# print("Accuracy of ShuffleSplit: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

def listToDict(lst):
    op = { i : lst[i] for i in range(0, len(lst) ) }
    return op


class ConfusionMatrixDisplay:

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):

        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots() #figsize=(40,40), dpi=100
            # ax.margins(x=-0.25, y=-0.25)
            # fig.set_size_inches(30, 30)
            # fig.figure(figsize=(20,10), dpi=100)
            # fig.f
            # fig.set_figheight(50)
            # fig.set_figwidth(50)
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color, fontsize=7)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")
        
        # ax.spines['right'].set_color('red')
        # ax.spines['bottom'].set_color('red')
        # ax.xaxis.label.set_color('red')
        # ax.tick_params(axis='x', colors='red')
        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(estimator, X, y_true, labels=None,sample_weight=None, normalize=None, display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None):
    check_matplotlib_support("plot_confusion_matrix")

    if not is_classifier(estimator):
        raise ValueError("plot_confusion_matrix only supports classifiers")

    y_pred = estimator.predict(X)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels, normalize=normalize)

    if display_labels is None:
        if labels is None:
        	display_labels = estimator.classes_
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(include_values=include_values, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation, values_format=values_format)


def run_algo(clf, X, y, algo_name):
	# cross_val(clf, X, y)
	class_names = ["https://www.google.com/","https://www.facebook.com/","https://www.heise.de/","https://www.b-tu.de/","https://www.amazon.com/","https://www.dotdash.com/","https://www.ebay.com/","http://www.spiegel.de/","https://www.bahn.com/i/view/index.shtml","https://en.wikipedia.org/wiki/Main_Page","https://twitter.com/","https://www.instagram.com/","https://www.linkedin.com/","https://www.flickr.com/","https://www.microsoft.com/de-de/","https://www.reddit.com/","https://www.nytimes.com/","https://www.jimdo.com/","https://www.blogger.com/","https://github.com/","https://www.mozilla.org/","https://myspace.com/","https://www.theguardian.com/","https://edition.cnn.com/us","https://www.paypal.com/","https://www.huffingtonpost.de/","https://www.dropbox.com/","https://www.xing.com/","https://www.owasp.org/index.php/Main_Page","https://www.skype.com/","https://www.springer.com/","https://www.oracle.com/index.html","https://www.android.com/","https://www.office.com/","https://www.alexa.com/","https://www.netflix.com/","https://opensource.org/","https://www.oracle.com/sun/index.html","https://www.snapchat.com/","https://www.symantec.com/","https://www.avira.com/","https://us.norton.com/","http://www.aboutads.info/","https://www.microsoft.com/","https://www.python.org/","https://siteorigin.com/","http://ewebdevelopment.com/","https://www.justice.gov/","https://www.warnerbros.com/","https://java.com/","https://www.playstation.com/","https://www.iso.org/home.html","https://moodle.org/","https://web.de/","https://www.gmx.net/","https://www.msn.com/","https://www.unicef.org/","https://www.nist.gov/","https://www.dmca.com/","http://taz.de/","http://www.faz.net/aktuell/","https://www.whatsapp.com/","https://www.samsung.com/us/","http://www.kicker.de/","https://translate.google.de/","https://www.transfermarkt.de/","https://www.sparkasse.de/","https://www.deutsche-bank.de/","https://www.ing-diba.de/","https://www.dkb.de/","https://www.commerzbank.de/","https://www.dhl.de/","https://www.wolframalpha.com/","https://www.history.de/","https://www.sueddeutsche.de/","https://www.bild.de/","https://www.chefkoch.de/","http://derbiokoch.de/","https://www.tagesschau.de/","https://www.wetteronline.de/","https://www.wetter.com/","https://prezi.com/","https://www.mindmeister.com/","https://www.notebooksbilliger.de/","https://de.napster.com/","https://www.deezer.com/de/","https://www.spotify.com/","https://soundcloud.com/","https://www.deutschlandradio.de/","http://www.surfmusik.de/","https://www.kino.de/","https://www.cinema.de/","https://www.imdb.com/","http://www.netzkino.de/","https://www.watchbox.de/","https://www.expedia.de/","https://www.opodo.de/"]
	class_names_new = ['de.napster.com', 'derbiokoch.de', 'edition.cnn.com', 'en.wikipedia.org', 'ewebdevelopment.com', 'github.com', 'java.com', 'moodle.org', 'myspace.com', 'opensource.org', 'prezi.com', 'siteorigin.com', 'soundcloud.com', 'taz.de', 'translate.google.de', 'twitter.com', 'us.norton.com', 'web.de', 'www.aboutads.info', 'www.alexa.com', 'www.amazon.com', 'www.android.com', 'www.avira.com', 'www.b-tu.de', 'www.bahn.com', 'www.bild.de', 'www.blogger.com', 'www.chefkoch.de', 'www.cinema.de', 'www.commerzbank.de', 'www.deezer.com', 'www.deutsche-bank.de', 'www.deutschlandradio.de', 'www.dhl.de', 'www.dkb.de', 'www.dmca.com', 'www.dotdash.com', 'www.dropbox.com', 'www.ebay.com', 'www.expedia.de', 'www.facebook.com', 'www.faz.net', 'www.flickr.com', 'www.gmx.net', 'www.google.com', 'www.heise.de', 'www.history.de', 'www.huffingtonpost.de', 'www.imdb.com', 'www.ing-diba.de', 'www.instagram.com', 'www.iso.org', 'www.jimdo.com', 'www.justice.gov', 'www.kicker.de', 'www.kino.de', 'www.linkedin.com', 'www.microsoft.com', 'www.mindmeister.com', 'www.mozilla.org', 'www.msn.com', 'www.netflix.com', 'www.netzkino.de', 'www.nist.gov', 'www.notebooksbilliger.de', 'www.nytimes.com', 'www.office.com', 'www.opodo.de', 'www.oracle.com', 'www.owasp.org', 'www.paypal.com', 'www.playstation.com', 'www.python.org', 'www.reddit.com', 'www.samsung.com', 'www.skype.com', 'www.snapchat.com', 'www.sparkasse.de', 'www.spiegel.de', 'www.spotify.com', 'www.springer.com', 'www.sueddeutsche.de', 'www.surfmusik.de', 'www.symantec.com', 'www.tagesschau.de', 'www.theguardian.com', 'www.transfermarkt.de', 'www.unicef.org', 'www.warnerbros.com', 'www.watchbox.de', 'www.wetter.com', 'www.wetteronline.de', 'www.whatsapp.com', 'www.wolframalpha.com', 'www.xing.com']

	kf = KFold(n_splits=10)
	index = 1
	scores = []
	for train_index, test_index in kf.split(X):
	    X_train, X_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
	    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

	    if(algo_name == "knn"):
	    	scaler = StandardScaler()
	    	scaler.fit(X_train)
	    	X_train = scaler.transform(X_train)
	    	X_test = scaler.transform(X_test)
	    clf.fit(X_train, y_train)
	    
	    curr_dir = os.getcwd()
	    filename = curr_dir + "/models/" + str(index) + '_' + algo_name + '_model.sav'

	    pickle.dump(clf, open(filename, 'wb'))

	    y_pred = clf.predict(X_test)
	    # print(y_pred)
	    conf_m_normalized = confusion_matrix(y_test, y_pred,  normalize="true", labels=class_names_new)
	    conf_m_non_normalized = confusion_matrix(y_test, y_pred, labels=class_names_new)

	    file_name_conf_norm = curr_dir + "/matrixes/" + str(index) + '_' + algo_name + '_conf_m_normalized.csv'
	    file_name_conf_not_norm = curr_dir + "/matrixes/" + str(index) + '_' + algo_name + '_conf_m_nonnormalized.csv'
	    
	    pd.DataFrame(conf_m_normalized).to_csv(file_name_conf_norm, header=None, index=None)
	    pd.DataFrame(conf_m_non_normalized).to_csv(file_name_conf_not_norm, header=None, index=None)

	    # df_X = pd.read_csv(file_name_conf_norm)
	    # df_y = pd.read_csv(file_name_conf_not_norm)

	    loaded_model = pickle.load(open(filename, 'rb'))
	    result = loaded_model.score(X_test, y_test)
	    
	    #print("Total accuracy of " + algo_name + ': ' + str(result))
	    scores.append(result)

	    # Plot non-normalized and normalized confusion matrixes
	    titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]
	    for title, normalize in titles_options:
	    	disp = plot_confusion_matrix( clf, X_test, y_test, 
	    		xticks_rotation = "vertical", display_labels=class_names,
	    		cmap=plt.cm.Blues,normalize=normalize)
	    	disp.ax_.set_title(title + " " + algo_name)
	    	disp.ax_.tick_params(axis='both', labelsize=8)

	    plt.show()

	    index += 1
	    # if index == 3:
	    # 	break
	print("Accuracy list of " + algo_name + ': ', scores)
	print("Overall accuracy: %0.2f (+/- %0.2f)" % (statistics.mean(scores), statistics.stdev(scores) * 2))


def svm(X, y, algo_name):
	clf = SVC(kernel='linear')
	run_algo(clf, X, y, algo_name)

def k_nearest_neighbor(X, y, algo_name):
	clf = KNeighborsClassifier(n_neighbors=5)
	run_algo(clf, X, y, algo_name)

def prepare_data(file_name):
	df = pd.read_csv(file_name)
	df = df.drop(columns="index")
	df_cumul_sum = df.iloc[[-1][0]]

	l = [0,10,20,30,40] #indexes for splitting
	list_of_dfs = [df_cumul_sum.iloc[l[n]:l[n+1]] for n in range(len(l)-1)] #splitting for four dataframes

	return df, df_cumul_sum, list_of_dfs

def data_preparation():
	#print(SUP.read_urls())
	urls_list = SUP.read_urls()

	joined_df = pd.DataFrame()
	for index, url in enumerate(urls_list):
		file_name = SUP.multi_files_names(url) 	#call multi_files_names() function to get specific file name for Marged Data
		
		path_to_file_name = SUP.merge_file_path(file_name) 				#call merge_file_path() function to enter into the directory of "file_name"
		
		df, df_cumul_sum, list_of_dfs = prepare_data(path_to_file_name) #bring necessary data in a proper format
		df = df.transpose()
		df.loc[0:40, 'site'] = url
		joined_df = joined_df.append(df) #ignore_index=True
		#print(df)
		# if index == 1:
		# 	break
	print(joined_df)
	WF.write_csv(joined_df, "joined_df" + file_name)
	return joined_df

def read_joined_df(file_name):
	path_to_file_name = SUP.merge_file_path_output(file_name)
	df = pd.read_csv(path_to_file_name)
	return df

def algorithms_selection(algo_name):
	#joined_df = data_preparation()
	#file_name = "joined_df_google_facebook.csv"

	file_name = "joined_df.csv"
	joined_df = read_joined_df(file_name)
	joined_df = joined_df.drop(columns="index")

	#print(joined_df.iloc[:,0:54].values.tolist())
	#print(joined_df.iloc[:,-1].values.tolist())
	X = joined_df.iloc[:,0:54].values.tolist()
	y = joined_df.iloc[:,-1].values.tolist()
	# print(np.isnan(X))
	# print(np.where(np.isnan(X)))
	X, y = unison_shuffled_copies(X, y)

	# pd.DataFrame(X).to_csv("X_data.csv", header=None, index=None)
	# pd.DataFrame(y).to_csv("y_data.csv", header=None, index=None)

	df_X = pd.read_csv('X_data.csv')
	df_y = pd.read_csv('y_data.csv')
	
	X = df_X.to_numpy()
	y = df_y.to_numpy()
	
	if(algo_name == "svm"):
		svm(X, y, algo_name)
	elif(algo_name == "knn"):
		k_nearest_neighbor(X, y, algo_name)
	elif(algo_name == "random_forest"):
		random_forest(X, y, algo_name)
	elif(algo_name == "neural_network"):
		neural_network(X, y, algo_name)


if __name__ == "__main__":
	#joined_df = data_preparation()
	#file_name = "joined_df_google_facebook.csv"
	file_name = "joined_df.csv"
	joined_df = read_joined_df(file_name)
	joined_df = joined_df.drop(columns="index")
	svm(joined_df)
	# k_nearest_neighbor(joined_df)
	# random_forest(joined_df)
	# neural_network(joined_df)

  
