import sys
import re
import matplotlib.pyplot as plt


def parse_results_file(filepath):

    seeds = []
    times = []
    average_time = None

    lines = None

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "latin-1"
    ]

    for encoding in encodings:

        try:
            with open(filepath, "r", encoding=encoding) as f:
                lines = f.readlines()

            print(f"[INFO] {filepath} opened with {encoding}")
            break

        except UnicodeDecodeError:
            continue

    if lines is None:
        raise Exception(f"Unable to read file: {filepath}")

    for line in lines:

        match = re.match(
            r"Seed\s+(\d+)\s*-\s*Total time:\s*([\d.]+)",
            line
        )

        if match:
            seed = int(match.group(1))
            time = float(match.group(2))

            seeds.append(seed)
            times.append(time)

        avg_match = re.match(
            r"Average total time:\s*([\d.]+)",
            line
        )

        if avg_match:
            average_time = float(avg_match.group(1))

    return seeds, times, average_time


def plot_results(file1, file2):

    seeds1, times1, avg1 = parse_results_file(file1)
    seeds2, times2, avg2 = parse_results_file(file2)

    if len(seeds1) == 0:
        print(f"Aucune donnée trouvée dans {file1}")
        return

    if len(seeds2) == 0:
        print(f"Aucune donnée trouvée dans {file2}")
        return

    if len(times1) != len(times2):
        print("Les deux fichiers n'ont pas le même nombre de résultats.")
        return

    colors2 = []

    nb_green = 0

    for t1, t2 in zip(times1, times2):

        if t2 > t1:
            colors2.append("red")

        elif t2 < t1:
            colors2.append("green")
            nb_green += 1

        else:
            colors2.append("blue")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.bar(seeds1, times1)
    ax1.axhline(avg1, linestyle="--", label=f"Average: {avg1:.2f}")

    ax1.set_title(file1)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Total time")

    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    ax2.bar(seeds2, times2, color=colors2)
    ax2.axhline(avg2, linestyle="--", label=f"Average: {avg2:.2f}")

    ax2.set_title(file2)
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Total time")

    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    print("\n==============================")
    print(f"{file1}")
    print(f"Average : {avg1:.2f}")
    print(f"Min     : {min(times1)}")
    print(f"Max     : {max(times1)}")

    print("\n==============================")
    print(f"{file2}")
    print(f"Average : {avg2:.2f}")
    print(f"Min     : {min(times2)}")
    print(f"Max     : {max(times2)}")
    print("==============================\n")

    print(f"Number of seeds where {file2} is better than {file1}: {nb_green} / {len(seeds1)}")

    plt.show()


def main():

    if len(sys.argv) != 3:
        print("Usage:")
        print("python plot_comparison.py file1.txt file2.txt")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    plot_results(file1, file2)


if __name__ == "__main__":
    main()