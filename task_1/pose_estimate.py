import numpy as np
import open3d as o3d
import os
import copy 
from sklearn.decomposition import PCA

# --- Configuration ---
DATA_DIR = "."  # Directory containing the files
INTRINSICS_FILE = os.path.join(DATA_DIR, "intrinsics.npy")
EXTRINSICS_FILE = os.path.join(DATA_DIR, "extrinsics.npy") 
DEPTH_FILE = os.path.join(DATA_DIR, "one-box.depth.npdata.npy")
COLOR_FILE = os.path.join(DATA_DIR, "one-box.color.npdata.npy")

# RANSAC Parameters (tune these based on noise level and scene)
RANSAC_DISTANCE_THRESHOLD = 0.005  # Max distance point to plane for inliers (meters)
RANSAC_N = 3                       # Min points to estimate plane
RANSAC_ITER = 10000                 # Number of iterations

def load_data():
    """Loads data files."""
    intrinsics = np.load(INTRINSICS_FILE)
    depth_map = np.load(DEPTH_FILE)
    color_map = np.load(COLOR_FILE) 

    # Normalize color map if it's not float 0-1
    if color_map.dtype == np.uint8:
        color_map = color_map / 255.0
    elif color_map.max() > 1.0: # Handle other potential scaling issues
        print("Warning: Color map values > 1.0. Normalizing.")
        color_map = color_map / color_map.max()

    return intrinsics, depth_map, color_map

def create_point_cloud_from_depth(depth_map, intrinsics, depth_scale=1.0, depth_trunc=3.0):
    """Creates an Open3D point cloud from a depth map and intrinsics."""
    height, width = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create Open3D Image objects
    o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))

    # Create point cloud from depth only
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy),
        depth_scale=depth_scale, depth_trunc=depth_trunc
    )
    return pcd

def segment_box_planes(pcd, min_plane_points=100):
    """Segments the dominant plane likely belonging to the box using RANSAC."""
    planes = []
    plane_inliers_pcds = []
    remaining_pcd = copy.deepcopy(pcd)

    # Since the camera is seeing from the top, we expect only one dominant plane
    if len(remaining_pcd.points) < RANSAC_N * 5:  # Need sufficient points
        print("Not enough points to find a plane.")
        return None, None

    # segment_plane returns: (a, b, c, d), list_of_indices
    plane_model, inlier_indices = remaining_pcd.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITER)

    if len(inlier_indices) < min_plane_points:
        print(f"Found plane with only {len(inlier_indices)} points. Stopping search.")
        return None, None

    print(f"Found dominant plane with {len(inlier_indices)} inliers. Normal: {plane_model[:3]}")
    planes.append(plane_model)

    # Select inliers and store their point cloud representation
    inlier_cloud = remaining_pcd.select_by_index(inlier_indices)
    plane_inliers_pcds.append(inlier_cloud)

    # Combine all inlier points into one cloud
    box_pcd = inlier_cloud
    
    # remove outliers from the box_pcd
    box_pcd, _ = box_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if not box_pcd.has_points():
        print("Error: No points associated with the detected plane.")
        return None, None

    print(f"Total points attributed to the dominant plane: {len(box_pcd.points)}")    
    return planes, box_pcd

def estimate_pose_from_planes(box_pcd):
    """Estimates Rotation (R) and Translation (t) from plane normals and points."""
    # --- Estimate Rotation (R) ---
    R = np.identity(3) # Default to identity
    # Use PCA to find the two main axes (x and y) for the plane

    # Extract points from the box point cloud
    points = np.asarray(box_pcd.points)

    # Apply PCA to the points
    pca = PCA(n_components=3)
    pca.fit(points)

    # The first two principal components represent the x and y axes
    x_axis = pca.components_[0]
    y_axis = pca.components_[1]

    # Ensure the axes are orthogonal
    z_axis = np.cross(x_axis, y_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # keep z negative so it always points up w.rt.camera frame (camera z-axis is negative, as camera looks down)
    if z_axis[2] > 0:
        z_axis = -z_axis

    # Construct the rotation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))

    # --- Estimate Translation (t) ---
    # Use the center of the combined inlier points
    center = box_pcd.get_center()
    t = center

    print(f"Estimated Rotation (R):\n{R}")
    print(f"Estimated Translation (t): {t}")

    # --- Construct 4x4 Transformation Matrix (Camera to Object) ---
    # This matrix transforms points FROM the object's frame TO the camera's frame
    T_cam_obj = np.identity(4)
    T_cam_obj[0:3, 0:3] = R
    T_cam_obj[0:3, 3] = t

    return T_cam_obj

def visualize_result(pcd, pose_matrix, box_pcd):
    """Visualizes the point cloud and the estimated bounding box."""
    if pose_matrix is None:
        print("No pose matrix to visualize.")
        o3d.visualization.draw_geometries([pcd])
        return

    # Create a 3D bounding box for the box point cloud
    bbox = box_pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)  # Red color for the bounding box
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    box_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    box_coord_frame.transform(pose_matrix)
    
    # set box_pcd color to green
    box_pcd.paint_uniform_color([0, 1, 0])
    
    # set pcd color to black
    pcd.paint_uniform_color([0.5, 0.5, 1])  
    
    # Visualize
    print("Visualizing Point Cloud (original) and Estimated Box Pose...")
    print("Press 'Q' or close the window to exit.")
    o3d.visualization.draw_geometries([box_pcd, pcd, bbox, box_coord_frame, coord_frame])

if __name__ == "__main__":
    # --- Main Execution ---
    # 1. Load Data
    intrinsics, depth_map, color_map = load_data()

    # 2. Create Point Cloud
    print("Creating point cloud...")
    pcd = create_point_cloud_from_depth(depth_map, intrinsics)
    # Applying statistical outlier rejection 
    # (num_neighbors, std_ratio) -  can tune these parameters
    pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(pcd_filtered.points)
    # Filter points with depth values between 0 and 2.5
    # This is a rough filter based on the expected height of the box
    filtered_indices = np.where((points[:, 2] >= 0) & (points[:, 2] <= 2.5))[0]
    filtered_points = points[filtered_indices]
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    display_pcd = filtered_pcd 

    # 3. Segment Box Planes using RANSAC
    print("Segmenting planes using RANSAC...")
    planes, box_pcd = segment_box_planes(display_pcd, min_plane_points=100) 

    # 4. Estimate Pose
    pose_matrix = estimate_pose_from_planes(box_pcd)

    # 5. Visualize
    # Visualize using the original *dense* point cloud for better context
    visualize_result(pcd, pose_matrix, box_pcd)

    # 6. Output the estimated pose matrix (if successful)
    print("\nFinal Estimated Pose (4x4 Transformation Matrix - Camera to Object Frame):")
    print(pose_matrix)
    
    # save the pose matrix to a npy file
    np.save(os.path.join(DATA_DIR, "estimated_pose.npy"), pose_matrix)
