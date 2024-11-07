import argparse
import os
import json
import matplotlib.pyplot as plt


def combine_plots(input_dirs, output_dir):

    # Verify the input directories exist
    input_dirs = input_dirs.split(",")
    for input_dir in input_dirs:
        path = os.path.join("results", input_dir, "results.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input directory {path} does not exist")

    # Read in the results from each input directory
    results = {}
    for input_dir in input_dirs:
        with open(os.path.join("results", input_dir, "results.json"), "r") as f:
            results[input_dir] = json.load(f)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot 1: Varying input dimensionality
    plt.subplot(1, 3, 1)
    for input_dir in input_dirs:
        dimensionality_list = results[input_dir]["experiments"][0][
            "dimensionality_list"
        ]
        cpu_times_dims = results[input_dir]["experiments"][0]["cpu_times_dims"]
        gpu_times_dims = results[input_dir]["experiments"][0]["gpu_times_dims"]
        plt.plot(
            dimensionality_list, cpu_times_dims, label=f"CPU ({input_dir})", marker="o"
        )
        if None not in gpu_times_dims:
            plt.plot(
                dimensionality_list,
                gpu_times_dims,
                label=f"GPU ({input_dir})",
                marker="o",
            )

        plt.xlabel("Input Dimensionality")
        plt.ylabel("Time per Epoch (s)")
        plt.title(
            f"""Time per Epoch vs Input Dimensionality\n"""
            f"""({results[input_dir]['experiments'][0]['num_points_fixpoint']} points, """
            f"""{results[input_dir]['experiments'][0]['num_epochs_fixpoint']} epochs)"""
        )
        plt.legend()

    # Plot 2: Varying number of points
    plt.subplot(1, 3, 2)
    for input_dir in input_dirs:
        num_points_list = results[input_dir]["experiments"][1]["num_points_list"]
        dimensionality_fixpoint = results[input_dir]["experiments"][1][
            "dimensionality_fixpoint"
        ]
        num_epochs_fixpoint = results[input_dir]["experiments"][1][
            "num_epochs_fixpoint"
        ]
        cpu_times_points = results[input_dir]["experiments"][1]["cpu_times_points"]
        gpu_times_points = results[input_dir]["experiments"][1]["gpu_times_points"]
        plt.plot(
            num_points_list, cpu_times_points, label=f"CPU ({input_dir})", marker="o"
        )
        if None not in gpu_times_points:
            plt.plot(
                num_points_list,
                gpu_times_points,
                label=f"GPU ({input_dir})",
                marker="o",
            )
        plt.xlabel("Number of Points")
        plt.ylabel("Time per Epoch (s)")
        plt.title(
            f"""Time per Epoch vs Number of Points\n"""
            f"""({dimensionality_fixpoint} dimensions, {num_epochs_fixpoint} epochs)"""
        )
        plt.legend()

    # Plot 3: Varying number of epochs
    plt.subplot(1, 3, 3)
    for input_dir in input_dirs:
        num_epochs_list = results[input_dir]["experiments"][2]["num_epochs_list"]
        dimensionality_fixpoint = results[input_dir]["experiments"][2][
            "dimensionality_fixpoint"
        ]
        num_points_fixpoint = results[input_dir]["experiments"][2][
            "num_points_fixpoint"
        ]
        cpu_times_epochs = results[input_dir]["experiments"][2]["cpu_times_epochs"]
        gpu_times_epochs = results[input_dir]["experiments"][2]["gpu_times_epochs"]
        plt.plot(
            num_epochs_list, cpu_times_epochs, label=f"CPU ({input_dir})", marker="o"
        )
        if None not in gpu_times_epochs:
            plt.plot(
                num_epochs_list,
                gpu_times_epochs,
                label=f"GPU ({input_dir})",
                marker="o",
            )
        plt.xlabel("Number of Epochs")
        plt.ylabel("Time per Epoch (s)")
        plt.title(
            f"""Time per Epoch vs Number of Epochs\n"""
            f"""({dimensionality_fixpoint} dimensions, {num_points_fixpoint} points)"""
        )
        plt.legend()

    os.makedirs(os.path.join("results", output_dir), exist_ok=True)
    plt.savefig(os.path.join("results", output_dir, "combined_plots.png"))

    # Write a json of the combined results
    with open(os.path.join("results", output_dir, "combined_results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine plots")
    parser.add_argument(
        "--inputs", type=str, help="Input directory names, comma-separated"
    )
    parser.add_argument("--output", type=str, help="Output directory name")
    args = parser.parse_args()
    combine_plots(args.inputs, args.output)
