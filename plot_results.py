import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


CSV_FILE = "controlled_density_edge_flip_results.csv"

X_COL_NUM = 14  # 1-based column number for density_delta_from_avg

SYSTEM_MATRIX_TYPE_COL_NUM = 2  # 1-based column number for system matrix type
GRAPH_TYPE_COL_NUM = 3  # 1-based column number for graph_type

ALLOWED_SYSTEM_MATRIX_TYPES = {"adjacency", "neg_laplacian"}
ALLOWED_GRAPH_TYPES = {"connected_ER", "connected_RG", "BA"}


def validate_filter_values(filter_by, filter_values):
    """
    Validate requested filter values.

    Parameters
    ----------
    filter_by : str
        Either "graph_type" or "system_matrix_type".
    filter_values : list[str], set[str], or None
        Requested filter values. If None, all valid values for filter_by are used.

    Returns
    -------
    set[str]
        Set of filter values to include.
    """
    if filter_by == "graph_type":
        allowed_values = ALLOWED_GRAPH_TYPES
    elif filter_by == "system_matrix_type":
        allowed_values = ALLOWED_SYSTEM_MATRIX_TYPES
    else:
        raise ValueError(
            f"Invalid filter_by value: {filter_by}. "
            "Allowed values are: graph_type, system_matrix_type"
        )

    if filter_values is None:
        return allowed_values

    filter_values = set(filter_values)
    invalid_filter_values = filter_values - allowed_values

    if invalid_filter_values:
        raise ValueError(
            f"Invalid {filter_by} value(s): {sorted(invalid_filter_values)}. "
            f"Allowed values are: {sorted(allowed_values)}"
        )

    return filter_values


def clean_latex_label(label):
    """
    Convert labels like '$\\rho_S$' into '$|\\rho_S|$'.
    """
    label = label.strip()

    # Remove surrounding $ signs if present
    if label.startswith("$") and label.endswith("$"):
        label = label[1:-1]

    return rf"$|{label}|$"


def extract_column_data(csv_file, y_col_num, filter_by="graph_type", filter_values=None):
    """
    Extract x = 14th column and y = user-selected column.

    Rows are filtered either by graph_type from column 3 or by
    system_matrix_type from column 2.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    y_col_num : int
        1-based column number to plot on y-axis.
    filter_by : str
        Either "graph_type" or "system_matrix_type".
    filter_values : list[str], set[str], or None
        Values to include for the chosen filter. If None, all valid values
        for filter_by are included.

    Returns
    -------
    x_vals : np.ndarray
        Values from density_delta_from_avg.
    y_vals_raw : np.ndarray
        Raw y-values from selected column.
    y_vals_abs : np.ndarray
        Absolute y-values for plotting.
    legend_label : str
        Matplotlib label based on previous column label.
    """
    filter_values = validate_filter_values(filter_by, filter_values)

    x_idx = X_COL_NUM - 1
    y_idx = y_col_num - 1
    label_idx = y_idx - 1

    if filter_by == "graph_type":
        filter_idx = GRAPH_TYPE_COL_NUM - 1
    elif filter_by == "system_matrix_type":
        filter_idx = SYSTEM_MATRIX_TYPE_COL_NUM - 1
    else:
        raise ValueError(
            f"Invalid filter_by value: {filter_by}. "
            "Allowed values are: graph_type, system_matrix_type"
        )

    if y_col_num <= 1:
        raise ValueError(
            "y_col_num must be greater than 1 because the legend label "
            "is taken from the previous column."
        )

    x_vals = []
    y_vals_raw = []
    labels = []

    with open(csv_file, newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) <= max(x_idx, filter_idx, y_idx, label_idx):
                continue

            filter_value = row[filter_idx].strip()

            if filter_value not in filter_values:
                continue

            try:
                x = float(row[x_idx])
                y = float(row[y_idx])
            except ValueError:
                # Skips header row or rows where selected column is not numeric
                continue

            x_vals.append(x)
            y_vals_raw.append(y)
            labels.append(row[label_idx].strip())

    if not x_vals:
        raise ValueError(
            f"No numeric data found for y-column {y_col_num} "
            f"with {filter_by} filter {sorted(filter_values)}. "
            "Check that the chosen column is numeric and that matching rows exist."
        )

    print(
        f"Found {len(x_vals)} data points for column {y_col_num} "
        f"with {filter_by} filter {sorted(filter_values)}."
    )

    legend_label = clean_latex_label(labels[0])

    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals_raw, dtype=float),
        np.abs(np.asarray(y_vals_raw, dtype=float)),
        legend_label,
    )


def make_scatter_plot(
    csv_file,
    y_col_nums,
    save_filename=None,
    filter_by="graph_type",
    filter_values=None,
    thresholds=None,
):
    """
    Make scatter plots for multiple selected y-columns.

    Each selected column is plotted in a separate subplot row.
    Rows can be filtered either by graph_type or by system_matrix_type.

    Different filter values are shown using different marker shapes and colors.
    If filter_by == "graph_type", shapes/colors correspond to graph types.
    If filter_by == "system_matrix_type", shapes/colors correspond to matrix types.

    Legend entries are shown only once in a single row at the top of all plots.

    For each selected column, filter value, and threshold, this function finds
    the plotted data point with the maximum x-axis value among points whose
    absolute coefficient value is below the threshold. It then draws a vertical
    dashed line at that x-axis value using the same color as the corresponding
    filter's scatter plot.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    y_col_nums : list[int]
        1-based column numbers to plot.
    save_filename : str or None
        If provided, save the figure to this filename.
    filter_by : str
        Either "graph_type" or "system_matrix_type".
    filter_values : list[str], set[str], or None
        Values to include for the chosen filter. If None, all valid values
        for filter_by are included.
    thresholds : list[float], tuple[float], or None
        Thresholds used to draw vertical reference lines. For each filter value
        and threshold, a line is drawn at the maximum x-value among plotted data
        points whose absolute coefficient value is below the threshold. If None,
        defaults to [0.95, 0.8].
    """
    filter_values = validate_filter_values(filter_by, filter_values)
    if thresholds is None:
        thresholds = [0.95, 0.8]
    thresholds = [float(threshold) for threshold in thresholds]
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")

    graph_type_markers = {
        "connected_ER": "o",
        "connected_RG": "s",
        "BA": "^",
    }

    graph_type_colors = {
        "connected_ER": "tab:blue",
        "connected_RG": "tab:orange",
        "BA": "tab:purple",
    }

    system_matrix_type_markers = {
        "adjacency": "o",
        "neg_laplacian": "s",
    }

    system_matrix_type_colors = {
        "adjacency": "tab:blue",
        "neg_laplacian": "tab:red",
    }

    if filter_by == "graph_type":
        filter_markers = graph_type_markers
        filter_colors = graph_type_colors
    elif filter_by == "system_matrix_type":
        filter_markers = system_matrix_type_markers
        filter_colors = system_matrix_type_colors
    else:
        raise ValueError(
            f"Invalid filter_by value: {filter_by}. "
            "Allowed values are: graph_type, system_matrix_type"
        )

    n_plots = len(y_col_nums)

    y_size_factor = 1.5
    if n_plots == 1:
        y_size_factor = 2.0

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(3, y_size_factor * n_plots),
        sharex=True,
    )

    # Ensure axes is iterable for the single-subplot case
    if n_plots == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for ax_idx, (ax, y_col_num) in enumerate(zip(axes, y_col_nums)):
        print(f"\nSelected y-column number: {y_col_num}")
        print(f"Selected filter mode: {filter_by}")
        print(f"Selected {filter_by} filter: {sorted(filter_values)}")
        print(f"Selected threshold(s): {thresholds}")

        column_max_x_values_below_threshold = {
            threshold: [] for threshold in thresholds
        }

        ylabel = None

        for filter_value in sorted(filter_values):
            x_vals, y_vals_raw, y_vals_abs, ylabel = extract_column_data(
                csv_file,
                y_col_num,
                filter_by=filter_by,
                filter_values=[filter_value],
            )

            if filter_by == "graph_type":
                legend_label = filter_value.replace("connected_", "")
            else:
                legend_label = filter_value

            print(f"\n{filter_by}: {filter_value}")
            print("First 10 raw values of chosen y-column:")
            print(y_vals_raw[:10])

            for threshold in thresholds:
                below_threshold_mask = y_vals_abs < threshold
                x_vals_below_threshold = x_vals[below_threshold_mask]

                if len(x_vals_below_threshold) > 0:
                    max_x_below_threshold = np.max(x_vals_below_threshold)
                    column_max_x_values_below_threshold[threshold].append(
                        max_x_below_threshold
                    )
                    print(
                        f"Maximum x-axis value with coefficient < "
                        f"{threshold}: {max_x_below_threshold}"
                    )
                    ax.axvline(
                        max_x_below_threshold,
                        color=filter_colors[filter_value],
                        linestyle="--",
                        linewidth=1,
                    )
                else:
                    print(f"No points found with coefficient < {threshold}")

            scatter = ax.scatter(
                x_vals,
                y_vals_abs,
                s=7,
                marker=filter_markers[filter_value],
                color=filter_colors[filter_value],
                label=legend_label,
            )

            # Add legend entries only once
            if ax_idx == 0:
                legend_handles.append(scatter)
                legend_labels.append(legend_label)

        print(f"\nOverall summary for y-column {y_col_num}:")
        for threshold in thresholds:
            if len(column_max_x_values_below_threshold[threshold]) > 0:
                overall_max_x = np.max(
                    column_max_x_values_below_threshold[threshold]
                )
                print(
                    f"Overall maximum x-axis value with coefficient < "
                    f"{threshold}: {overall_max_x}"
                )
            else:
                print(
                    f"Overall: no points found with coefficient < {threshold}"
                )

        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("Density range")

    # One shared legend in a single row at the top of all plots
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_labels),
        fontsize="small",
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved plot to: {save_filename}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot abs(selected CSV columns) vs density_delta_from_avg."
    )

    parser.add_argument(
        "y_col_nums",
        type=int,
        nargs="+",
        help="1-based column numbers to plot on the y-axis.",
    )

    parser.add_argument(
        "--csv",
        default=CSV_FILE,
        help=f"CSV filename. Default: {CSV_FILE}",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Filename to save the plot as a PNG file.",
    )

    parser.add_argument(
        "--graph_type",
        nargs="+",
        choices=sorted(ALLOWED_GRAPH_TYPES),
        default=None,
        help=(
            "Graph type(s) to include. "
            "Allowed values: connected_ER, connected_RG, BA. "
            "Default: include all graph types unless --system_matrix_type is used."
        ),
    )

    parser.add_argument(
        "--system_matrix_type",
        nargs="+",
        choices=sorted(ALLOWED_SYSTEM_MATRIX_TYPES),
        default=None,
        help=(
            "System matrix type(s) to include. "
            "Allowed values: adjacency, neg_laplacian. "
            "If provided, colors and shapes correspond to system matrix type."
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.95, 0.8],
        help=(
            "Threshold(s) for vertical reference lines. For each filter value "
            "and threshold, a line is drawn at the maximum x-value among plotted "
            "data points whose absolute y-value is below the threshold. "
            "Default: 0.95 0.8."
        ),
    )

    args = parser.parse_args()

    if args.graph_type is not None and args.system_matrix_type is not None:
        raise ValueError(
            "Use either --graph_type or --system_matrix_type, not both."
        )

    if args.system_matrix_type is not None:
        filter_by = "system_matrix_type"
        filter_values = args.system_matrix_type
    else:
        filter_by = "graph_type"
        filter_values = args.graph_type

    make_scatter_plot(
        args.csv,
        args.y_col_nums,
        save_filename=args.save_name,
        filter_by=filter_by,
        filter_values=filter_values,
        thresholds=args.thresholds,
    )