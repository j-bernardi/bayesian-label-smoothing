import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: values, errors, title
def add_to_bar_plot(ax, vals, data_col, error_col):
    pos = np.arange(len(vals))
    ax.bar(
        pos,
        vals[data_col],
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


def plot_from_preprepared():

    # NOTE this csv contains manual processing.
    # creating combine.csv from results.csv procedurally is a TODO
    x = pd.read_csv("results/smoothing/combine.csv")
    # Take the pre
    fig, ax = plt.subplots(3, 2)
    add_to_bar_plot(ax[0, 0], x, 'Total accuracy', 'stdev')
    add_to_bar_plot(ax[0, 1], x, 'Total accuracy', 'range')

    add_to_bar_plot(ax[1, 0], x, 'avg_cls_acc', 'stdev')
    add_to_bar_plot(ax[1, 1], x, 'avg_cls_acc', 'range')

    add_to_bar_plot(ax[2, 0], x, 'avg_cls_acc_exc_bg', 'stdev')
    add_to_bar_plot(ax[2, 1], x, 'avg_cls_acc_exc_bg', 'range')


def csv_to_dict(csvfile="results.csv", printing=True):
    """Convert main.py results to experiment dict

    Takes the output csv file of main.py and converts it
    to a dict. Reads experiments in row format:
        path/to/expName_N  val1  val2  ...

    Returns:
        dict of: {
            expName: {
                val1: {mean, std, range, raw},
                val2: {mean, std, range, raw},
            },
            ...
        }
    """

    # Cols: exp, acc, avg_cls_acc, avg_cls_acc_exc_bg, val_loss
    all_results = pd.read_csv(csvfile)

    # {exp1: {quantity1: {}, ...}, ...}
    result_compilations = {}

    raw_exp_names = all_results["exp"]
    remove_dirs = raw_exp_names.str.split("/", expand=True).iloc[:, -1]
    remove_repeats = remove_dirs.str.split("_", expand=True).iloc[:, :-1]
    exp_names = remove_repeats.agg("_".join, axis=1)

    # Condense all columns that start with exp_name
    for exp in exp_names:
        exp_repeats = all_results[ all_results["exp"].str.contains(exp) ]
        if exp not in result_compilations:
            result_compilations[exp] = {}

        for col in [c for c in exp_repeats.columns if c != "exp"]:
            result_compilations[exp][col] = {
                "mean": np.mean(exp_repeats[col]),
                "std": np.std(exp_repeats[col]),
                "range": (
                    np.min(exp_repeats[col]),
                    np.median(exp_repeats[col]),
                    np.max(exp_repeats[col])
                ),
                "raw": exp_repeats[col].values,
            }
    if printing:
        pprint.pprint(result_compilations)
    return result_compilations


def plot_sweep_on_scatter(result_dict, quantity="acc"):
    """Plot the sweep of parameters for each experiment

    Take a result dict output by csv_to_dict, collect each
    experiment series (i.e. parameter sweep), and plot
    them as a scatter graph with error bars.
    """

    # COLLECT EXPERIMENTS
    series = {}

    for exp in result_dict:

        series_name = "_".join(exp.split("_")[:-1])
        param_value = float("0." + exp.split("_")[-1])

        if series_name not in series:
            series[series_name] = {
                "x_vals": [],
                "y_vals": [],
                "err_vals": [],
            }

        # if mean, std (TODO: else range[0, 1, 2])
        series[series_name]["x_vals"].append(param_value)
        series[series_name]["y_vals"].append(
            result_dict[exp][quantity]["mean"]
        )
        series[series_name]["err_vals"].append(
            result_dict[exp][quantity]["std"]
        )

    # PLOT
    plt.figure()
    legend = []
    for series_name in series:
        legend.append(series_name)
        plt.errorbar(
            series[series_name]["x_vals"],
            series[series_name]["y_vals"],
            yerr=series[series_name]["err_vals"],
            capsize=10.0,
        )
    plt.legend(legend)
    plt.title(f"{quantity} mean, std")  # VARY
    plt.xlabel("Parameter value")
    plt.ylabel(quantity)
    return series


if __name__ == "__main__":

    result_dict = csv_to_dict("results/smoothing_optim/fixed_results.csv")
    # pickle save?
    for quant in ["acc", "avg_cls_acc", "avg_cls_acc_exc_bg"]:
        plot_sweep_on_scatter(result_dict, quant)
    plt.show()

    # Plot from prepared CSV (to be deprecated)
    # plot_from_preprepared()
