# Imports
import os
import h5py
import numpy as np # Needed for NaN
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import itertools # For combinations of objects

# --- Publication-quality style settings ---
# (Keep the settings as they were)
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Title font size for individual axes
    'axes.labelsize': 16,     # Font size for x and y labels
    'xtick.labelsize': 14,    # Font size for x-axis tick labels
    'ytick.labelsize': 14,    # Font size for y-axis tick labels
    'legend.fontsize': 14,    # Font size for the legend
    'figure.titlesize': 20,   # Overall figure title font size (if used)
    'lines.linewidth': 2,     # Default line width
    'lines.markersize': 6,    # Default marker size (adjust as needed)
    'grid.alpha': 0.5,        # Transparency of grid lines
    'grid.linestyle': '--'    # Style of grid lines
})


def load_tracking_data(hdf5_path):
    """
    Loads tracking data from the HDF5 file into a dictionary structure.
    (Implementation from previous version - assumed correct)
    """
    print(f"Loading data from: {hdf5_path}")
    data = defaultdict(lambda: defaultdict(dict))
    metadata = {}
    try:
        with h5py.File(hdf5_path, 'r') as hf:
            metadata['fps'] = hf.attrs.get('fps', 30.0)
            metadata['original_width'] = hf.attrs.get('original_width')
            metadata['original_height'] = hf.attrs.get('original_height')
            metadata['processing_width'] = hf.attrs.get('processing_width')
            metadata['processing_height'] = hf.attrs.get('processing_height')
            metadata['num_frames'] = hf.attrs.get('num_frames')
            metadata['total_objects_prompted'] = hf.attrs.get('total_objects_prompted')
            print(f"  Metadata: FPS={metadata['fps']:.2f}, OrigDims=({metadata['original_width']}x{metadata['original_height']}), NumFrames={metadata['num_frames']}")
            if 'frames' not in hf: print("Error: 'frames' group not found."); return None, None
            frames_group = hf['frames']
            frame_keys = sorted([key for key in frames_group.keys() if key.startswith('frame_')], key=lambda x: int(x.split('_')[1]))
            print(f"  Found {len(frame_keys)} processed frame groups.")
            for frame_key in frame_keys:
                try:
                    frame_idx = int(frame_key.split('_')[1])
                    frame_group = frames_group[frame_key]
                    for item_key in frame_group.keys():
                        if item_key.endswith('_mask') and item_key.startswith('object_'):
                            obj_dataset = frame_group[item_key]
                            try:
                                obj_id = obj_dataset.attrs.get('global_id')
                                class_name = obj_dataset.attrs.get('class_name', 'Unknown')
                                centroid_x = obj_dataset.attrs.get('centroid_x')
                                centroid_y = obj_dataset.attrs.get('centroid_y')
                                bbox = obj_dataset.attrs.get('bbox_xyxy')
                                if obj_id is not None:
                                    if centroid_x is not None and centroid_y is not None: data[obj_id][frame_idx]['centroid'] = (centroid_x, centroid_y)
                                    if bbox is not None: data[obj_id][frame_idx]['bbox'] = list(bbox)
                                    if frame_idx not in data[obj_id]: data[obj_id][frame_idx] = {}
                                    data[obj_id][frame_idx]['class'] = class_name
                            except Exception as e_attr: print(f"Warning: Error reading attributes for {item_key} in {frame_key}: {e_attr}")
                except Exception as e_frame: print(f"Warning: Error processing frame group {frame_key}: {e_frame}")
    except FileNotFoundError: print(f"Error: HDF5 file not found at {hdf5_path}"); return None, None
    except Exception as e_open: print(f"Error opening or reading HDF5 file {hdf5_path}: {e_open}"); return None, None
    if not data: print("Warning: No tracking data extracted.")
    print(f"Data loaded for {len(data)} objects.")
    return data, metadata


def calculate_movement(tracking_data, fps):
    """
    Calculates total distance traveled and average speed for each object.
    (Implementation from previous version - assumed correct)
    """
    movement_stats = {}
    if not tracking_data or fps <= 0: print("Warning: Cannot calculate movement."); return movement_stats
    time_interval = 1.0 / fps
    for obj_id, frame_data in tracking_data.items():
        total_distance = 0.0; total_time_elapsed = 0.0
        sorted_frames = sorted(frame_data.keys()); frames_tracked = len(sorted_frames); frames_with_centroid = 0
        if frames_tracked < 2: movement_stats[obj_id] = {'total_distance': 0.0, 'average_speed': 0.0, 'frames_tracked': frames_tracked, 'frames_with_centroid': 0}; continue
        last_centroid = None; last_frame_idx_with_centroid = -1
        for frame_idx in sorted_frames:
            current_info = frame_data[frame_idx]; current_centroid = current_info.get('centroid')
            if current_centroid is not None:
                frames_with_centroid += 1
                if last_centroid is not None:
                    dx = current_centroid[0] - last_centroid[0]; dy = current_centroid[1] - last_centroid[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    frame_diff = frame_idx - last_frame_idx_with_centroid; time_diff = frame_diff * time_interval
                    total_distance += distance; total_time_elapsed += time_diff
                last_centroid = current_centroid; last_frame_idx_with_centroid = frame_idx
        avg_speed = total_distance / total_time_elapsed if total_time_elapsed > 0 else 0.0
        movement_stats[obj_id] = {'total_distance': total_distance, 'average_speed': avg_speed, 'frames_tracked': frames_tracked, 'frames_with_centroid': frames_with_centroid}
        print(f"  Obj {obj_id}: Dist={total_distance:.2f}px, AvgSpeed={avg_speed:.2f}px/s, Tracked={frames_tracked}f ({frames_with_centroid} usable)")
    return movement_stats


def calculate_inter_object_distances(tracking_data, obj_id1, obj_id2):
    """
    Calculates the Euclidean distance between the centroids of two specified objects.
    (Implementation from previous version - assumed correct)
    """
    distances = {}
    if not tracking_data: print("Warning: No tracking data provided."); return distances
    if obj_id1 not in tracking_data: print(f"Warning: Object ID {obj_id1} not found."); return distances
    if obj_id2 not in tracking_data: print(f"Warning: Object ID {obj_id2} not found."); return distances
    frames1 = set(tracking_data[obj_id1].keys()); frames2 = set(tracking_data[obj_id2].keys())
    common_frames = sorted(list(frames1.intersection(frames2)))
    valid_distance_points = 0
    for frame_idx in common_frames:
        if frame_idx not in tracking_data[obj_id1] or frame_idx not in tracking_data[obj_id2]: continue
        info1 = tracking_data[obj_id1][frame_idx]; info2 = tracking_data[obj_id2][frame_idx]
        centroid1 = info1.get('centroid'); centroid2 = info2.get('centroid')
        if centroid1 is not None and centroid2 is not None:
            dx = centroid1[0] - centroid2[0]; dy = centroid1[1] - centroid2[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances[frame_idx] = distance; valid_distance_points += 1
    print(f"Calculated distance between Obj {obj_id1} and Obj {obj_id2} for {valid_distance_points} frames.")
    if valid_distance_points == 0 and len(common_frames) > 0: print(f"  (Objects coexisted in {len(common_frames)} frames, but centroids were missing).")
    return distances


# --- MODIFIED plot_trajectories Function ---
def plot_trajectories(tracking_data, metadata, output_dir="."):
    """
    Plots the centroid trajectories, avoiding connections to the origin (0,0).
    """
    if not tracking_data:
        print("No tracking data provided for plotting trajectories.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    width = metadata.get('original_width')
    height = metadata.get('original_height')

    print("Generating trajectory plot (ignoring origin connections)...")
    plotted_ids = []

    for obj_id, frame_data in tracking_data.items():
        frames = sorted(frame_data.keys())
        frames_with_centroids = [f for f in frames if 'centroid' in frame_data[f]]

        # Extract coordinates into numpy arrays
        x_coords_np = np.array([frame_data[f]['centroid'][0] for f in frames_with_centroids])
        y_coords_np = np.array([frame_data[f]['centroid'][1] for f in frames_with_centroids])

        if x_coords_np.size == 0: # Skip if no valid centroids
            continue

        # --- Modification Start ---
        # Identify points at the origin (0,0)
        # Adding a small epsilon in case of floating point inaccuracies near zero
        epsilon = 1e-6
        is_origin = (np.abs(x_coords_np) < epsilon) & (np.abs(y_coords_np) < epsilon)

        # Create copies to modify, ensuring float type for NaN
        plot_x = x_coords_np.copy().astype(float)
        plot_y = y_coords_np.copy().astype(float)

        # Replace origin points with NaN
        plot_x[is_origin] = np.nan
        plot_y[is_origin] = np.nan
        # --- Modification End ---

        # Only plot if there are non-NaN points left
        if not np.all(np.isnan(plot_x)):
            plotted_ids.append(obj_id)
            color = plt.get_cmap('tab10')(obj_id % 10)

            # Plot the trajectory line (will have breaks at NaNs)
            ax.plot(plot_x, plot_y, marker='o', linestyle='-', color=color,
                    markersize=4, linewidth=1.5, label=f'Obj {obj_id}')

            # Mark start point only if it wasn't the origin
            if not is_origin[0]:
                ax.scatter(x_coords_np[0], y_coords_np[0], marker='s', s=80,
                           facecolors='none', edgecolors=color, linewidth=1.5, label=f'_Start {obj_id}')

            # Mark end point only if it wasn't the origin
            if not is_origin[-1]:
                ax.scatter(x_coords_np[-1], y_coords_np[-1], marker='X', s=100,
                           color=color, label=f'_End {obj_id}')

    # --- Plot Customization ---
    ax.set_title('Centroid Trajectories of Tracked Objects')
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    if width is not None and height is not None:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect('equal', 'box')
    else:
        print("Warning: Original dimensions not found. Plot may auto-scale.")
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
    ax.grid(True)

    if plotted_ids:
        if len(plotted_ids) > 10:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            ax.legend(loc='best', ncol=2, frameon=True)
            plt.tight_layout()
    else:
        print("Warning: No objects with valid (non-origin) trajectories were plotted.")
        plt.tight_layout()

    # --- Save Figure ---
    plot_filename = os.path.join(output_dir, 'object_trajectories_pub_no_origin_lines.png') # New name
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Trajectory plot saved for publication to: {plot_filename}')
    except Exception as e:
        print(f"Error saving trajectory plot: {e}")
    plt.close(fig)


def plot_inter_object_distance(distances, obj_id1, obj_id2, fps, output_dir="."):
    """
    Plots the distance between two objects over time for publication.
    (Implementation from previous version - assumed correct)
    """
    if not distances:
        print(f"No valid distance data calculated for Obj {obj_id1} vs {obj_id2}. Skipping plot.")
        return
    frames = sorted(distances.keys()); time_vals = [f / fps for f in frames]; dist_vals = [distances[f] for f in frames]
    if not dist_vals: print(f'No valid distance data points found to plot for Obj {obj_id1} vs {obj_id2}.'); return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_vals, dist_vals, marker='.', linestyle='-', linewidth=2, markersize=5)
    ax.set_title(f'Distance Between Obj {obj_id1} and Obj {obj_id2} Over Time')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Distance (pixels)')
    ax.grid(True); plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'distance_obj{obj_id1}_obj{obj_id2}_pub.png')
    try: plt.savefig(plot_filename, dpi=300, bbox_inches='tight'); print(f'Distance plot saved to: {plot_filename}')
    except Exception as e: print(f"Error saving distance plot: {e}")
    plt.close(fig)


# --- Main Analysis Execution ---
if __name__ == '__main__':
    # (Argument parsing and main workflow logic remains the same as previous version)
    parser = argparse.ArgumentParser(description="Analyze tracking data from HDF5 file.")
    parser.add_argument("hdf5_file", help="Path to the input HDF5 tracking file.")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to save plots and CSV files (default: current directory).")
    parser.add_argument("--obj1", type=int, help="First object ID for distance analysis.")
    parser.add_argument("--obj2", type=int, help="Second object ID for distance analysis.")
    parser.add_argument("--plot", action='store_true', help="Generate trajectory and distance plots.")
    parser.add_argument("--save_csv", action='store_true', help="Save extracted statistics and distances to CSV files.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    tracking_data, metadata = load_tracking_data(args.hdf5_file)

    if tracking_data and metadata:
        fps = metadata.get('fps', 30.0)
        if fps <= 0: print("Warning: Invalid FPS. Using default 30.0."); fps = 30.0

        # 2. Calculate Movement Stats
        print("\n--- Movement Statistics ---")
        movement_stats = calculate_movement(tracking_data, fps)
        if args.save_csv and movement_stats:
            try:
                movement_df = pd.DataFrame.from_dict(movement_stats, orient='index')
                movement_df.index.name = 'object_id'
                csv_path = os.path.join(args.output_dir, "movement_statistics.csv")
                movement_df.to_csv(csv_path)
                print(f"Movement statistics saved to: {csv_path}")
            except Exception as e: print(f"Could not save movement statistics to CSV: {e}")

        # 3. Calculate Inter-Object Distance
        distances = None
        if args.obj1 is not None and args.obj2 is not None:
            print(f"\n--- Inter-Object Distance (Obj {args.obj1} vs Obj {args.obj2}) ---")
            distances = calculate_inter_object_distances(tracking_data, args.obj1, args.obj2)
            count = 0
            for frame, dist in distances.items():
                 if count < 5: print(f"  Frame {frame}: Distance = {dist:.2f} px")
                 count += 1
            if count == 0: print("  No frames found where both objects had valid centroids.")
            elif count > 5: print(f"  ... ({count} total valid distance points)")
            if args.save_csv and distances:
                try:
                    dist_df = pd.DataFrame.from_dict(distances, orient='index', columns=['distance_pixels'])
                    dist_df.index.name = 'frame_index'
                    dist_csv_path = os.path.join(args.output_dir, f"distance_obj{args.obj1}_obj{args.obj2}.csv")
                    dist_df.to_csv(dist_csv_path)
                    print(f"Distance data saved to: {dist_csv_path}")
                except Exception as e: print(f"Could not save distance data to CSV: {e}")

        # 4. Plotting
        if args.plot:
            print("\n--- Plotting Trajectories ---")
            plot_trajectories(tracking_data, metadata, args.output_dir) # Call modified version
            if distances is not None:
                 plot_inter_object_distance(distances, args.obj1, args.obj2, fps, args.output_dir)
            elif args.obj1 is not None or args.obj2 is not None:
                 print("\nSkipping inter-object distance plot (requires both --obj1 and --obj2).")

        print("\nAnalysis complete.")
    else:
        print("Failed to load data. Exiting analysis.")
