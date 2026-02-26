import os
import csv
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def consolidate_br_rewards(br_rewards_dir, output_filename):
    """
    Consolidates BR rewards from br_rewards directory. For each main checkpoint step:
    - adv files: take the most negative (worst from exploiter-as-adversary's point of view)
    - ego files: take the most positive (best from exploiter-as-ego's point of view)

    File pattern: {checkpoint}_br{index}_adv.txt and {checkpoint}_br{index}_ego.txt
    """
    stats_adv = {}  # checkpoint -> min (most negative) adv reward
    stats_ego = {}  # checkpoint -> max (most positive) ego reward

    if not os.path.exists(br_rewards_dir):
        print(f"Directory not found: {br_rewards_dir}")
        return

    # Pattern: {checkpoint}_br{index}_adv.txt or {checkpoint}_br{index}_ego.txt
    pattern = re.compile(r"^(\d+)_br\d+_(adv|ego)\.txt$")

    for filename in os.listdir(br_rewards_dir):
        match = pattern.match(filename)
        if not match:
            continue
        checkpoint = int(match.group(1))
        file_type = match.group(2)
        file_path = os.path.join(br_rewards_dir, filename)
        try:
            with open(file_path, "r") as f:
                value = float(f.read().strip())
            if file_type == "adv":
                if checkpoint not in stats_adv or value > stats_adv[checkpoint]:
                    stats_adv[checkpoint] = value
            else:  # ego
                if checkpoint not in stats_ego or value < stats_ego[checkpoint]:
                    stats_ego[checkpoint] = value
        except (ValueError, IOError) as e:
            print(f"Could not process file: {filename} - {e}")

    # Merge checkpoints (union of keys from both)
    all_checkpoints = sorted(set(stats_adv.keys()) | set(stats_ego.keys()))

    if all_checkpoints:
        try:
            with open(output_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Checkpoint", "WorstAdvReward", "BestEgoReward"])
                for cp in all_checkpoints:
                    adv_val = stats_adv.get(cp, "")
                    ego_val = stats_ego.get(cp, "")
                    writer.writerow([cp, adv_val, ego_val])
            print(f"Successfully wrote {output_filename} with {len(all_checkpoints)} checkpoints.")
        except IOError as e:
            print(f"Error writing to {output_filename}: {e}")


def consolidate_selfplay_rewards(selfplay_rewards_dir, output_filename, value_header="MeanReward"):
    """
    Consolidates selfplay rewards from selfplay_rewards directory. For each main checkpoint step,
    reads all files matching {checkpoint}_br{index}.txt and takes the average (same selfplay model
    regardless of br_index).

    File pattern: {checkpoint}_br{index}.txt
    """
    stats = defaultdict(list)

    if not os.path.exists(selfplay_rewards_dir):
        print(f"Directory not found: {selfplay_rewards_dir}")
        return

    # Pattern: {checkpoint}_br{index}.txt
    pattern = re.compile(r"^(\d+)_br\d+\.txt$")

    for filename in os.listdir(selfplay_rewards_dir):
        match = pattern.match(filename)
        if not match:
            continue
        checkpoint = int(match.group(1))
        file_path = os.path.join(selfplay_rewards_dir, filename)
        try:
            with open(file_path, "r") as f:
                value = float(f.read().strip())
            stats[checkpoint].append(value)
        except (ValueError, IOError) as e:
            print(f"Could not process file: {filename} - {e}")

    if stats:
        sorted_stats = sorted((cp, np.mean(vals)) for cp, vals in stats.items())
        try:
            with open(output_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Checkpoint", value_header])
                writer.writerows(sorted_stats)
            print(f"Successfully wrote {output_filename} with {len(sorted_stats)} checkpoints (averaged).")
        except IOError as e:
            print(f"Error writing to {output_filename}: {e}")


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

def plot_br_csv_data(csv_path, base_filename=None):
    """
    Reads a BR CSV file with Checkpoint, WorstAdvReward, BestEgoReward and plots both series.
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) < 3:
            print(f"Error: BR CSV must have Checkpoint, WorstAdvReward, BestEgoReward")
            return
        x_col = df.columns[0]
        # Drop rows where either value is missing for plotting
        df_plot = df.replace("", np.nan).dropna(subset=[df.columns[1], df.columns[2]], how="all")
        if df.columns[1] in df_plot.columns and df_plot[df.columns[1]].notna().any():
            plt.plot(df_plot[x_col], pd.to_numeric(df_plot[df.columns[1]], errors="coerce"), marker="o", linestyle="-", label="WorstAdvReward")
        if df.columns[2] in df_plot.columns and df_plot[df.columns[2]].notna().any():
            plt.plot(df_plot[x_col], pd.to_numeric(df_plot[df.columns[2]], errors="coerce"), marker="s", linestyle="-", label="BestEgoReward")
        plt.title("BR: worst-case adv / best-case ego (exploiter perspective)")
        plt.xlabel(x_col)
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        base_filename = os.path.splitext(os.path.basename(csv_path))[0] if base_filename is None else base_filename
        output_filename = f"{base_filename}_plot.png"
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
    except Exception as e:
        print(f"An error occurred while processing '{csv_path}': {e}")


def plot_csv_data(csv_path, base_filename=None):
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
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')

        # Add titles and labels
        plt.title("Exploiter performance compared to belief on performance")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        
        # Save the plot
        base_filename = os.path.splitext(os.path.basename(csv_path))[0] if base_filename is None else base_filename
        output_filename = f"{base_filename}_plot.png"
        plt.savefig(output_filename)

        print(f"Plot saved to {output_filename}")
        #plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
    except Exception as e:
        print(f"An error occurred while processing '{csv_path}': {e}")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    br_mean_rew_stats_dir = os.path.join(current_dir, "br_rewards")
    selfplay_mean_rew_stats_dir = os.path.join(current_dir, "selfplay_rewards")

    this_moment = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

    output_br_mean_rew_file = os.path.join(current_dir, f"br_mean_rewards_{this_moment}.csv")
    output_selfplay_mean_rew_file = os.path.join(current_dir, f"selfplay_mean_rewards_{this_moment}.csv")
    #consolidate_stats(wr_stats_dir, output_wr_file, "WinRate")
    consolidate_br_rewards(br_mean_rew_stats_dir, output_br_mean_rew_file)
    consolidate_selfplay_rewards(selfplay_mean_rew_stats_dir, output_selfplay_mean_rew_file, "MeanReward")
    br_base_filename = "br_mean_rewards_%s" % this_moment
    plt.figure(figsize=(10, 6))
    plot_br_csv_data(output_br_mean_rew_file, base_filename=br_base_filename)
    selfplay_base_filename = "selfplay_mean_rewards_%s" % this_moment
    plt.figure(figsize=(10, 6))
    plot_csv_data(output_selfplay_mean_rew_file, base_filename=selfplay_base_filename)
    # Combined plot: BR (WorstAdv, BestEgo) + Selfplay
    plt.figure(figsize=(10, 6))
    df_br = pd.read_csv(output_br_mean_rew_file)
    x_col = df_br.columns[0]
    for col in ["WorstAdvReward", "BestEgoReward"]:
        if col in df_br.columns:
            valid = pd.to_numeric(df_br[col], errors="coerce")
            plt.plot(df_br[x_col], valid, marker="o", linestyle="-", label=f"BR {col}")
    df_sp = pd.read_csv(output_selfplay_mean_rew_file)
    plt.plot(df_sp[df_sp.columns[0]], df_sp[df_sp.columns[1]], marker="s", linestyle="-", label="Selfplay")
    plt.xlabel("Checkpoint")
    plt.ylabel("Reward")
    plt.title("BR (worst adv / best ego) vs Selfplay")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(current_dir, f"br_vs_selfplay_mean_rewards_{this_moment}.png"))
    print(f"Combined plot saved to {os.path.join(current_dir, f'br_vs_selfplay_mean_rewards_{this_moment}.png')}")