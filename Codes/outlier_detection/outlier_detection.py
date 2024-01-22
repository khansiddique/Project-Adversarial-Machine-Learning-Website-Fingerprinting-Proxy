from scapy.all import *
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import support_functions as SUP
from collections import defaultdict
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from numpy import array
import write_file as WF
import log_dataset_csv as csv
import time
from memory_profiler import memory_usage
import math
import outliers_evaluation as EV
import outlier_detection as DET
from sklearn.ensemble import IsolationForest
import random
from random import shuffle

def isolation_forest(file_name, path_to_file_name, is_Simul, isMax, max_number_or_var, min_number):
	df, df_cumul_sum, list_of_dfs = prepare_data(path_to_file_name) #bring necessary data in a proper format
	#print(df)
	df_removed = df #in future here will be stored adjsted dataframe
	outliers_list = [] #outliers
	number_of_outliers = 0

	map_groups = group_column_names(df)
	for i, group in enumerate(map_groups):
		if group != 'index':
			joined_df = pd.DataFrame()
			for index, name in enumerate(map_groups[group]):
				new_df = df[["index", name]]
				new_df.columns = ['index', 'y']
				joined_df = joined_df.append(new_df, ignore_index=True)
			print(joined_df)
			X_train = pd.DataFrame(joined_df, columns = ['index', 'y'])
			clf = IsolationForest(contamination = 0.02, random_state = 0)
			clf.fit(X_train)
			y_pred_train = clf.predict(X_train)
			print(list(y_pred_train))
			print("Accuracy:", list(y_pred_train).count(1)/y_pred_train.shape[0])

			number_of_outlier_points = list(y_pred_train).count(-1)
			print(number_of_outlier_points)
			mean_number_of_outlier_points = number_of_outlier_points / 10
			crawling_id = 0
			outliers_indexes = []
			for index, i in enumerate(y_pred_train):
				if i == -1:
					outliers_indexes.append(index)
				if (index + 1) % 50 == 0:
					outliers_from_subpage = {}
					if mean_number_of_outlier_points < len(outliers_indexes):
						outliers_from_subpage[map_groups[group][crawling_id]] = outliers_indexes
						outliers_list.append(outliers_from_subpage)
					outliers_indexes = []
					crawling_id += 1
			#print(outliers_list)
		
		#df_checked = (list_of_dfs[index] < (Q1 - 1.5 * IQR)) | (list_of_dfs[index] > (Q3 + 1.5 * IQR))
		#df_outliers = list_of_dfs[index][((list_of_dfs[index] < (Q1 - 1.5 * IQR)) |(list_of_dfs[index] > (Q3 + 1.5 * IQR)))]  #dataframe of outliers
		#print(df_outliers)
		#df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list = upper_bound_check(is_Simul, df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list)
	# if(is_Simul == False):
	# 	WF.write_csv(df_removed, "iqr_removed_" + file_name)
	print("Outliers based on IsolationForest:")
	print_comparison("IS", path_to_file_name, outliers_list, number_of_outliers, df, df_removed, len(df_cumul_sum.index)) #print information


def IQR(file_name, path_to_file_name, is_Simul, isMax, max_number_or_var, min_number):
	df, df_cumul_sum, list_of_dfs = prepare_data(path_to_file_name) #bring necessary data in a proper format

	df_removed = df #in future here will be stored adjsted dataframe
	outliers_list = [] #outliers based on final cumul sum
	number_of_outliers = 0
	for index in range(len(list_of_dfs)): #go through dataframe of each page
		Q1 = list_of_dfs[index].quantile(0.25)
		Q3 = list_of_dfs[index].quantile(0.75)
		IQR = Q3 - Q1
		#df_checked = (list_of_dfs[index] < (Q1 - 1.5 * IQR)) | (list_of_dfs[index] > (Q3 + 1.5 * IQR))
		df_outliers = list_of_dfs[index][((list_of_dfs[index] < (Q1 - 1.5 * IQR)) |(list_of_dfs[index] > (Q3 + 1.5 * IQR)))]  #dataframe of outliers
		#print(df_outliers)

		df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list = upper_bound_check(is_Simul, df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list)
	# print(outliers_list)
	df_removed, list_of_random_outliers = upper_rand_check(outliers_list, df, max_number_or_var)
	if(is_Simul == False):
		WF.write_csv(df_removed, "iqr_removed_" + file_name)
	print("Outliers based on IQR based on final CUMUL_SUM:")
	print_comparison("IQR_CUMUL_SUM", path_to_file_name, outliers_list, number_of_outliers, df, df_removed, len(df_cumul_sum.index)) #print information

def z_score(file_name, path_to_file_name, is_Simul, isMax, max_number_or_var, min_number):
    df, df_cumul_sum, list_of_dfs = prepare_data(path_to_file_name) #bring necessary data in a proper format

    df_removed = df #in future here will be stored adjsted dataframe
    outliers_list = [] #outliers based on final cumul sum
    threshold = 1.8
    number_of_outliers = 0
    for index in range(len(list_of_dfs)): #go through dataframe of each page
    	#print(list_of_dfs[index])
    	mean = np.mean(list_of_dfs[index]) #mean from instances of one page
    	std = np.std(list_of_dfs[index]) #std from instances of one page
    	df_outliers = list_of_dfs[index][np.abs(list_of_dfs[index] - mean)/std > threshold] #dataframe of outliers based on z = (X — μ) / σ
    	
    	df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list = upper_bound_check(is_Simul, df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list)
    if(is_Simul):
    	WF.write_csv(df_removed, "z-score_removed_" + file_name)
    print("Outliers based on Z-score based on final CUMUL_SUM:")
    print_comparison("Z-score_CUMUL_SUM", path_to_file_name, outliers_list, number_of_outliers, df, df_removed, len(df_cumul_sum.index)) #print information

def upper_rand_check(outliers_list, df, num_to_select):
	all_outliers = []
	for dictionary in outliers_list:
		for j in dictionary:
			all_outliers.append(j)
	#print(all_outliers)

	shuffle(all_outliers)
	counter = 0;
	list_of_random_outliers = []
	for outlier in all_outliers:
		if(num_to_select > counter):
			list_of_random_outliers.append(outlier)
		counter += 1

	print("Removed outliers: ", list_of_random_outliers)
	for i in list_of_random_outliers:
		df = df.drop(i, 1)
	#print(df)
	return df, list_of_random_outliers

def upper_bound_check(is_Simul, df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list):
	if(df_outliers.empty == False): #if any outliers exists
		outliers_from_subpage = {}
		for i in df_outliers.index:
			if(is_Simul == False):
				if(isMax):
					# if(max_number_or_var <= number_of_outliers):
					# # 	#print("number_of_outliers: ", number_of_outliers)
					# 	break
					pass
				else:
					if(int(max_number_or_var) >= int(measuring_variance(df_removed))):
						break
			df_removed = df_removed.drop(i, 1)
			outliers_from_subpage[i] = df_outliers[i]
			number_of_outliers += 1
		outliers_list.append(outliers_from_subpage)
	return df_outliers, isMax, max_number_or_var, number_of_outliers, df_removed, outliers_list

#bring necessary data in a proper format
def prepare_data(file_name):
	df = pd.read_csv(file_name)
	df = df.drop(columns="index")
	df_cumul_sum = df.iloc[[-1][0]]

	l = [0,10,20,30,40] #indexes for splitting
	list_of_dfs = [df_cumul_sum.iloc[l[n]:l[n+1]] for n in range(len(l)-1)] #splitting for four dataframes

	return df, df_cumul_sum, list_of_dfs

#print information about algorithm
def print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features):
	#print(df_removed)
	print(outliers_list)
	#number_of_outliers = [sum(v != 0 for v in i) for i in outliers_list]
	print("Number of outliers: ", number_of_outliers)
	print("Number of all features: ", number_of_all_features)
	before_dist = measuring_sum_dist(df)
	after_dist = measuring_sum_dist(df_removed)

	before_var = measuring_variance(df)
	after_var = measuring_variance(df_removed)
	# print("Mean variance before removing: ", before_var)
	# print("Mean variance after removing: ", after_var)
	var_improvements_percentage = ((before_var - after_var) / before_var) * 100
	print("Mean variance reduced by: ", var_improvements_percentage, "%")
	average_efficiency = 0
	if number_of_outliers != 0:
		average_efficiency = var_improvements_percentage / number_of_outliers
	print("Variance Average efficiency: 1 Outlier = ", average_efficiency, "%")

	# print("Sum of euclidean distance between instances of each page before removing: ", before_dist)
	# print("Sum of euclidean distance between instances of each page after removing: ", after_dist)
	dist_improvements_percentage = ((before_dist - after_dist) / before_dist) * 100
	print("Mean of euclidean distance reduced by: ", dist_improvements_percentage, "%")
	average_efficiency = 0
	if number_of_outliers != 0:
		average_efficiency = dist_improvements_percentage / number_of_outliers
	print("Distance Average efficiency: 1 Outlier = ", average_efficiency, "%")
	


	comp_list = [[algo_name, number_of_outliers, number_of_all_features, before_dist, after_dist, dist_improvements_percentage, average_efficiency]]
	#print(comp_list)
	csv.append_csv_file(comp_list, path_to_file_name + ".ori")
	
	print()

	#plotOutliers(algo_name, df, outliers_list)

def group_column_names(df):
	header_list = list(df.columns) #column names
	maps = defaultdict(list)  #datastructure for grouping instances based on pages
	for s in header_list:
		s_elements = s.split("_") # Here feature column Header is splited
		maps[s_elements[0]].append(s) #add propriate instance appropriately to group
	#print(maps)
	return maps

def measuring_sum_dist(df): #calculate the sum of sub_sums (distance between instances of one page)
	maps = group_column_names(df)
	
	number_of_distances = 0
	sum_distance = 0 #the sum of sub_sums
	dist = 0 #distance between instances of one page
	list_distance = [] #contains distances between instances of each page
	for index0, i in enumerate(maps): #go through each group
		#print(maps[i])
		l = len(maps[i])
		number_of_distances += l * (l - 1) / 2
		for index1, i_name in enumerate(maps[i]): #calculate euclidean distances between instances of each page
			for index2, j_name in enumerate(maps[i]):
				if(index1 < index2):
					dist += distance.euclidean(df[i_name], df[j_name]) #calculate and add distance
		list_distance.append(dist)
		sum_distance += dist
		dist = 0
	if(number_of_distances == 0):
		number_of_distances = 1
	sum_distance /= number_of_distances
	return sum_distance

def measuring_variance(df): #calculate the sum of the mean variance (distance between instances of one page)
	maps = group_column_names(df)

	number_of_distances = 0
	sum_distance = 0 #the sum of sub_sums
	dist = 0 #distance between instances of one page
	list_distance = [] #contains distances between instances of each page
	for index0, i in enumerate(maps): #go through each group
		#print(maps[i])
		l = len(maps[i])
		#print("LEN: ", l)
		number_of_distances += l * (l - 1) / 2
		for index1, i_name in enumerate(maps[i]): #calculate euclidean distances between instances of each page
			for index2, j_name in enumerate(maps[i]):
				if(index1 < index2):
					df_new = pd.DataFrame((df[i_name] - df[j_name]) * (df[i_name] - df[j_name]), columns = ['square_difference']) 
					#df_new["square_difference"] =  #calculate and add variance
					dist += df_new["square_difference"].sum()
		list_distance.append(dist)
		sum_distance += dist
		dist = 0
	#sum_distance /= len(df.columns())
	#sum_distance = math.sqrt(sum_distance)
	if(number_of_distances == 0):
		number_of_distances = 1
	sum_distance /= number_of_distances#df.shape[1]
	#print(sum_distance)
	#print("SDVSDDDDDDDDDDDD: ", number_of_distances)
	return sum_distance

def prepare_lits_for_groups(df):
	df = df.transpose()
	list_of_lists = df.iloc[:41].values
	groups_of_pages = []
	temp_list = []
	for index, sublist in enumerate(list_of_lists):
		temp_list.extend(sublist)
		if(index % 10 == 9):
			groups_of_pages.append(temp_list)
			temp_list = []

	return groups_of_pages

def get_prediction_from_linear_regression(page_list):
	x = np.array(list(np.arange(1, 51))*10).reshape((-1, 1))
	y = array(page_list)
	
	model = LinearRegression().fit(x, y)
	print('coefficient of determination:', model.score(x, y))
	print('intercept:', model.intercept_)
	print('slope:', model.coef_)

	y_pred = model.predict(np.arange(50).reshape((-1, 1)))
	#print('predicted response:', y_pred, sep='\n')

	# plt.scatter(np.arange(1, 51), y_pred, color='red')
	# plt.scatter(list(np.arange(1, 51))*10, y, color='blue')
	# plt.title('page name', fontsize=14)
	# plt.xlabel('Index', fontsize=14)
	# plt.ylabel('Packet size in Kb', fontsize=14)
	# plt.grid(True)
	# plt.show()

	return y_pred

def linear_regression(file_name, path_to_file_name, is_Simul):
	df, df_new, list_of_dfs = prepare_data(path_to_file_name)
	df_removed = df

	outliers_list = [] #outliers based on final cumul sum
	number_of_outliers = 0

	maps = group_column_names(df)

	groups_of_pages = prepare_lits_for_groups(df)
	group_distance_list = []
	dist = 0
	#print(groups_of_pages)
	for index, page_list in enumerate(groups_of_pages):
		y_pred = get_prediction_from_linear_regression(page_list)
		#measuring_dist(df)
		#print(y_pred)
		#df = df['prediction'].assign(e=pd.Series(array(y_pred)))
		df_with_pred = df.assign(prediction=pd.Series(y_pred).values)
		#print(maps[str(index+1)])
		page_distance_list = {}
		for index0, i_name in enumerate(maps[str(index+1)]):
			#print(i_name)
			#print(maps[str(index+1)][index0])
			d = distance.euclidean(df[i_name], df_with_pred['prediction'])
			page_distance_list[i_name] = d
		group_distance_list.append(page_distance_list)

	#print(group_distance_list)
	outlier_list = giveOutliers(group_distance_list)
	plotOutliers("linear_regression", df, outlier_list)
	# print("Outliers based on Linear_Regression")
	# print_comparison(outliers_list, number_of_outliers, df, df_removed, len(df_new.index)) #print information

def giveOutliers(group_distance_list):
	outlier_list = []
	for index, group in enumerate(group_distance_list):
		#print(sorted(group)
		t = list(group.values())
		print()
		# sorted_x = sorted(group.items(), key=lambda kv: kv[1])
		# print(sorted_x)
		q1, q3 = np.percentile(t,[25,75])
		iqr = q3 - q1
		lower_bound = q1 -(1.5 * iqr)
		upper_bound = q3 +(1.5 * iqr)
		print(lower_bound, " ", upper_bound)
		page_outliers_list = {}
		for i in t:
			if((i < lower_bound) | (i > upper_bound)):
				for k, v in group.items():
					if v == i:
						page_outliers_list[k] = i
		if len(page_outliers_list) != 0:
			outlier_list.append(page_outliers_list)
	print(outlier_list)
	return outlier_list


def plotOutliers(algo_name, df, outlier_list):
	fig, ax = plt.subplots(1,1,figsize=(16,7), dpi= 80)

	ax.set_xlabel('Index', fontsize=20)
	ax.set_ylabel('Packet size in Kb', color='tab:red', fontsize=20)
	ax.tick_params(axis='y', rotation=0, labelcolor='tab:red' )

	#colors = ['green', 'red', 'blue', 'orange', 'black', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
	colors = [['rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'salmon', 'tomato'], 
	['darkorange', 'burlywood', 'orange', 'darkgoldenrod', 'goldenrod', 'gold', 'khaki', 'darkkhaki', 'olive', 'yellow'],
	['lawngreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'springgreen'],
	['lightseagreen', 'mediumturquoise', 'darkslategrey', 'teal', 'aqua', 'lightblue', 'deepskyblue', 'steelblue', 'dodgerblue', 'lightslategray']]
	files_list = []
	for index1 in range(1, len(df.columns)):    
		file = []
		for index2 in range(1, len(df.columns)):
			if (df.columns[index1].split('_')[0] == df.columns[index2].split('_')[0]):
				file.append(df.columns[index2])
		files_list.append(file)
	files_list.sort()

	all_lists = list(files_list for files_list,_ in itertools.groupby(files_list))
	for group_index, group_webpage in enumerate(all_lists):
		for webpage_index, webpage in enumerate(group_webpage):
			if(check(webpage, outlier_list)):
				plt.plot(np.arange(0, 50, 1), df[webpage] / 1000.0, 'o', label = webpage, color=colors[group_index % len(colors)][webpage_index % 10])
			else:
				plt.plot(np.arange(0, 50, 1), df[webpage] / 1000.0, '-', label = webpage, color=colors[group_index % len(colors)][webpage_index % 10])
		plt.title(all_lists[0][0].split('_')[2] + " " + algo_name, fontsize=20)
		plt.legend()
		plt.show()

	# plt.scatter(np.arange(1, 51), y_pred, color='red')
	# plt.scatter(list(np.arange(1, 51))*10, y, color='blue')
	# plt.title('page name', fontsize=14)
	# plt.xlabel('Index', fontsize=14)
	# plt.ylabel('Packet size in Kb', fontsize=14)
	# plt.grid(True)
	# plt.show()

def check(webpage, outlier_list):
	for i in outlier_list:
		if i.get(webpage, 0) != 0:
				return True
	return False


def find_curves(file_name, path_to_file_name):
	df = pd.read_csv(path_to_file_name)
	df = df.drop(columns="index")
	df_cumul_sum = df.iloc[[-1][0]]
	#print(df_cumul_sum)
	min_curve_value = df_cumul_sum.min()
	max_curve_value = df_cumul_sum.max()
	mean_curve_value = df_cumul_sum.mean()
	params = df_cumul_sum.iloc[(df_cumul_sum-mean_curve_value).abs().argsort()[:1]]

	print(f"min curve name: {df_cumul_sum.idxmin()}")
	print(f"min curve value: {min_curve_value}")

	print(f"max curve name: {df_cumul_sum.idxmax()}")
	print(f"max curve value: {max_curve_value}")

	print(f"mean curve name: {params.index.values[0]}")
	print(f"mean curve value: {params.iloc[-1]}")
	print()


def simulation_mode(url_of_input_file, output_file):
 	file_name = SUP.multi_files_names(url_of_input_file) 	#call multi_files_names() function to get specific file name for Marged Data
 	path_to_file_name = SUP.merge_file_path(file_name) 				#call merge_file_path() function to enter into the directory of "file_name"
 	
 	header = ["Algo_name", "Number of outliers", "Number of all features", "before_dist", "after_dist", "dist_improvements_percentage", "Average efficiency"]
 	csv.write_header_csv_file(header, path_to_file_name + ".ori")

 	find_curves(file_name, path_to_file_name)

 	IQR_time_start = time.clock()
 	IQR(file_name, path_to_file_name, True, False, -1, -1)
 	IQR_time_finished = time.clock()
 	IQR_time = IQR_time_finished - IQR_time_start
 	#print("Time: ", IQR_time)

 	z_time_start = time.clock()
 	z_score(file_name, path_to_file_name, True, False, -1, -1)
 	z_time_finished = time.clock()
 	z_time = z_time_finished - z_time_start
 	#print("Time: ", z_time)

 	lr_time_start = time.clock()
 	linear_regression(file_name, path_to_file_name, True)
 	lr_time_finished = time.clock()
 	lr_time = lr_time_finished - lr_time_start
 	#print("Time: ", lr_time)

 	#print("MEMORY: " , memory_usage())
 	#print("MEMORY: " , memory_usage((IQR, (file_name,path_to_file_name, True))))
 	
 	#print(path_to_file_name)


def algorithm_running(algorithm_name, min_number_to_remove, max_number_to_remove, variance, input_file):
	if(max_number_to_remove != -1):
		invoke_function(algorithm_name, min_number_to_remove, max_number_to_remove, True, input_file)
	else:
		invoke_function(algorithm_name, min_number_to_remove, variance, False, input_file)

def invoke_function(algorithm_name, min_number_to_remove, max_number_or_var, isMax, input_file):
	file_name = SUP.multi_files_names(input_file) 	#call multi_files_names() function to get specific file name for Marged Data
	path_to_file_name = SUP.merge_file_path(file_name) 		#call merge_file_path() function to enter into the directory of "file_name"

	if(algorithm_name == "Z-score_CUMUL_SUM"):
		z_score(file_name, path_to_file_name, False, isMax, max_number_or_var, min_number_to_remove)
	elif(algorithm_name == "IQR_CUMUL_SUM"):
		IQR(file_name, path_to_file_name, False, isMax, max_number_or_var, min_number_to_remove)
	elif(algorithm_name == "Z-score_ALL_POINTS"):
		algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_Z_score(input_file, path_to_file_name, False)
		DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
	elif(algorithm_name == "IQR_ALL_POINTS"):
		algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_IQR(input_file, path_to_file_name, False)
		DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
	elif(algorithm_name == "LR_ALL_POINTS"):
		algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_LR(input_file, path_to_file_name, False)
		DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
	elif(algorithm_name == "LR_10_40"):
		linear_regression()
	elif(algorithm_name == "IS"):
		isolation_forest(file_name, path_to_file_name, False, isMax, max_number_or_var, min_number_to_remove)


if __name__ == "__main__":
	simulation_mode('www.google.com', 'www.google.com.ori')

  
