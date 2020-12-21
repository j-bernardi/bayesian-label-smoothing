import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("results/smoothing/combine.csv")

# print(x.columns)
# print(x['Total accuracy'])

def bar(ax, vals, data_col, error_col):
	pos = np.arange(len(vals))
	ax.bar(
		pos, vals[data_col],
		yerr=vals[error_col],
		align='center', alpha=0.5,
		ecolor='black', capsize=10,
	)
	ax.set_xticks(pos)
	ax.minorticks_on()
	ax.set_xticklabels(vals['Unnamed: 0'])
	ax.set_title(f"{data_col}, {error_col}")

	# Customize the major grid
	ax.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='black')
	# Customize the minor grid
	ax.grid(axis='y', which='minor', linestyle='--', linewidth='0.2', color='black')
	ax.set_ylabel('Accuracy (%)')
	ax.set_yticks(np.arange(0., 1., 0.05))
	ax.set_yticks(np.arange(0., 1., 0.01), minor=True)
	ax.set_ylim(0.6, 1.0 if max(vals[data_col]) > 0.8 else 0.8)

	# Save the figure and show
	plt.tight_layout()
	# plt.savefig('bar_plot_with_error_bars.png')

fig, ax = plt.subplots(3, 2)
bar(ax[0, 0], x, 'Total accuracy', 'stdev')
bar(ax[0, 1], x, 'Total accuracy', 'range')

bar(ax[1, 0], x, 'avg_cls_acc', 'stdev')
bar(ax[1, 1], x, 'avg_cls_acc', 'range')

bar(ax[2, 0], x, 'avg_cls_acc_exc_bg', 'stdev')
bar(ax[2, 1], x, 'avg_cls_acc_exc_bg', 'range')

plt.show()
