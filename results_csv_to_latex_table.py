import csv

input_file = "results_format_conversion/results2.csv"
output_file = "results_format_conversion/results2.tex"

name_map = {
    "connected_ER": ("ER", "p"),
    "connected_RG": ("RG", "d"),
    "BA": ("BA", "m"),
}

with open(input_file, newline="") as f, open(output_file, "w") as out:
    reader = csv.reader(f)

    for row in reader:
        if not row:
            continue

        graph_type = row[2].strip()
        param = row[4].strip()

        mean = row[5].strip()
        std = row[6].strip()

        if float(std) < 1e-10:
            std = "0"

        transform = row[7].strip().capitalize()

        # first correlation pair
        spearman1, spearman1_std = row[12].strip(), row[13].strip()
        kendall1, kendall1_std = row[15].strip(), row[16].strip()

        if float(spearman1_std) < 1e-10:
            spearman1_std = "0"
        if float(kendall1_std) < 1e-10:
            kendall1_std = "0"

        # third correlation pair
        spearman3, spearman3_std = row[28].strip(), row[29].strip()
        kendall3, kendall3_std = row[31].strip(), row[32].strip()

        if float(spearman3_std) < 1e-10:
            spearman3_std = "0"
        if float(kendall3_std) < 1e-10:
            kendall3_std = "0"

        short, param_name = name_map[graph_type]

        line = (
            f"{short} ({param_name} = {param}) & "
            f"{mean} $\\pm$ {std} & "
            # f"{transform} & "
            f"{spearman1} $\\pm$ {spearman1_std} & "
            f"{kendall1} $\\pm$ {kendall1_std} & "
            f"{spearman3} $\\pm$ {spearman3_std} & "
            f"{kendall3} $\\pm$ {kendall3_std} \\\\\n"
        )

        out.write(line)

print(f"Saved formatted rows to {output_file}")