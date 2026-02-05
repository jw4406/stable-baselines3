import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def consolidate_stats(stats_dir, output_filename, value_header):
    """
    Reads all .txt files from a directory, appends their data to a CSV file,
    and then deletes the .txt files. If the CSV file exists, it's updated
    with new data; otherwise, it's created.
    """
    stats = {}
    
    # 1. Read existing data from CSV if it exists
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    stats[int(row[0])] = float(row[1])
        except (IOError, StopIteration, ValueError, IndexError) as e:
            print(f"Warning: Could not read existing CSV {output_filename}. It might be empty or corrupted. Error: {e}")

    if not os.path.exists(stats_dir):
        print(f"Directory not found: {stats_dir}")
        return

    processed_files = []
    # 2. Read new data from .txt files
    for filename in os.listdir(stats_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(stats_dir, filename)
            try:
                checkpoint_num = int(os.path.splitext(filename)[0])
                with open(file_path, 'r') as f:
                    value = float(f.read().strip())
                stats[checkpoint_num] = value
                processed_files.append(file_path)
            except (ValueError, IndexError):
                print(f"Could not process file: {filename}")
    
    # 3. Write updated data back to CSV
    if stats:
        sorted_stats = sorted(stats.items())
        try:
            with open(output_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Checkpoint", value_header])
                writer.writerows(sorted_stats)
            
            # 4. Delete processed .txt files only after successful write
            # for file_path in processed_files:
            #     os.remove(file_path)

            print(f"Successfully updated {output_filename} and processed {len(processed_files)} source file(s).")
        except IOError as e:
            print(f"Error writing to {output_filename}. Source files were retained. Error: {e}")

def plot_csv_data(csv_path):
    """
    Reads a CSV file with two columns (Global Step and a value) and creates a plot.

    Args:
        csv_path (str): The path to the CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get column names
        if len(df.columns) < 2:
            print(f"Error: CSV file '{csv_path}' must have at least two columns.")
            return

        x_col = df.columns[0]
        y_col = df.columns[1]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')

        # Add titles and labels
        plt.title(f'{y_col} vs. {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        
        # Save the plot
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_filename}_plot.png"
        plt.savefig(output_filename)

        print(f"Plot saved to {output_filename}")
        plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
    except Exception as e:
        print(f"An error occurred while processing '{csv_path}': {e}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #wr_stats_dir = os.path.join(current_dir, "trained_models/wr_stats")
    mean_rew_stats_dir = os.path.join(current_dir, "rewards")
    this_moment = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    #output_wr_file = os.path.join(current_dir, "win_rates.csv")
    output_mean_rew_file = os.path.join(current_dir, f"mean_rewards_{this_moment}.csv")

    #consolidate_stats(wr_stats_dir, output_wr_file, "WinRate")
    consolidate_stats(mean_rew_stats_dir, output_mean_rew_file, "MeanReward")
    plot_csv_data(output_mean_rew_file)
