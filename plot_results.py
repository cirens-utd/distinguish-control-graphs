import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


CSV_FILE = "controlled_density_edge_flip_results.csv"
X_COL_NUM = 14  # 1-based column number for density_delta_from_avg


def clean_latex_label(label):
    """
    Convert labels like '$\\rho_S$' into '$|\\rho_S|$'.
    """
    label = label.strip()

    # Remove surrounding $ signs if present
    if label.startswith("$") and label.endswith("$"):
        label = label[1:-1]

    return rf"$|{label}|$"


def extract_column_data(csv_file, y_col_num):
    """
    Extract x = 14th column and y = user-selected column.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    y_col_num : int
        1-based column number to plot on y-axis.

    Returns
    -------
    x_vals : np.ndarray
        Values from density_delta_from_avg.
    y_vals_raw : np.ndarray
        Raw y-values from selected column.
    y_vals_abs : np.ndarray
        Absolute y-values for plotting.
    ylabel : str
        Matplotlib ylabel based on previous column label.
    """
    x_idx = X_COL_NUM - 1
    y_idx = y_col_num - 1
    label_idx = y_idx - 1

    if y_col_num <= 1:
        raise ValueError("y_col_num must be greater than 1 because the ylabel is taken from the previous column.")

    x_vals = []
    y_vals_raw = []
    labels = []

    with open(csv_file, newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) <= max(x_idx, y_idx, label_idx):
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
            f"No numeric data found for y-column {y_col_num}. "
            "Check that the chosen column is a numeric quantity column."
        )
    else:
        print(f"Found {len(x_vals)} data points.")

    ylabel = clean_latex_label(labels[0])

    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals_raw, dtype=float),
        np.abs(np.asarray(y_vals_raw, dtype=float)),
        ylabel,
    )


def make_scatter_plot(csv_file, y_col_num, save_filename=None):
    x_vals, y_vals_raw, y_vals_abs, ylabel = extract_column_data(csv_file, y_col_num)

    print(f"Selected y-column number: {y_col_num}")
    print("First 10 raw values of chosen y-column:")
    print(y_vals_raw[:10])

    # print("\nFirst 10 absolute values used for plotting:")
    # print(y_vals_abs[:10])

    plt.figure()
    plt.scatter(x_vals, y_vals_abs)
    plt.xlabel("density_delta_from_avg")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}")
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=300)
        print(f"\nSaved plot to: {save_filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot abs(selected CSV column) vs density_delta_from_avg."
    )
    parser.add_argument(
        "y_col_num",
        type=int,
        help="1-based column number to plot on the y-axis.",
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

    args = parser.parse_args()
    make_scatter_plot(args.csv, args.y_col_num, save_filename=args.save_name)