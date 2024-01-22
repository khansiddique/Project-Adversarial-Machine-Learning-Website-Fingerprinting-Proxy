import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

def diagram(file_name):
  #df = pd.read_csv("1_1_www.google.com_tls.csv")
  df = pd.read_csv(file_name)
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
      plt.plot(np.arange(0, 50, 1), df[webpage] / 1000.0, '-', label = webpage, color=colors[group_index % len(colors)][webpage_index % 10])
  plt.title(all_lists[0][0].split('_')[2], fontsize=20)
  plt.legend()
  plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# num_plots = 20

# # Have a look at the colormaps here and decide which one you'd like:
# # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
# colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

# # Plot several different functions...
# x = np.arange(10)
# labels = []
# for i in range(1, num_plots + 1):
#     plt.plot(x, i * x + 5 * i)
#     labels.append(r'$y = %ix + %i$' % (i, 5*i))

# # I'm basically just demonstrating several different legend options here...
# plt.legend(labels, ncol=4, loc='upper center', 
#            bbox_to_anchor=[0.5, 1.1], 
#            columnspacing=1.0, labelspacing=0.0,
#            handletextpad=0.0, handlelength=1.5,
#            fancybox=True, shadow=True)

# plt.show()
