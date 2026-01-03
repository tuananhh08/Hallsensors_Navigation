import numpy as np
import pandas as pd

#load input files
helical_df = pd.read_csv("Helical_points_coordinates.csv")
sensor_df = pd.read_csv("Sensors_pos.csv")

helical_pts = helical_df.values
sensor_pos = sensor_df.values
helical_xyz = helical_pts[:,:3]

sensor_center = sensor_pos.mean(axis=0)

#define ROI 
roi_width = 0.12     
roi_depth = 0.12     
roi_height = 0.07    

x_min = sensor_center[0] - roi_width / 2
x_max = sensor_center[0] + roi_width / 2

y_min = sensor_center[1] - roi_depth / 2
y_max = sensor_center[1] + roi_depth / 2

z_min = sensor_center[2]
z_max = sensor_center[2] + roi_height

#gen data in ROI
num_xy = 20
num_z = 15

x_vals = np.linspace(x_min, x_max, num_xy)
y_vals = np.linspace(y_min, y_max, num_xy)
z_vals = np.linspace(z_min, z_max, num_z)

data = []  

for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            data.append([x, y, z])

data = np.array(data)               

cos_alpha = np.ones((data.shape[0], 1))           
cos_beta  = np.ones((data.shape[0], 1))           

data = np.hstack([data, cos_alpha, cos_beta])   

columns = ["x", "y", "z", "cos_alpha", "cos_beta"]
out_df = pd.DataFrame(data, columns=columns)
out_df.to_csv("ROI_data.csv", index=False)

print("Data generated")

#normalize xyz for loss 
xyz = data[:, :3]  

xyz_mean = xyz.mean(axis=0, keepdims=True)
xyz_std  = xyz.std(axis=0, keepdims=True) + 1e-12

xyz_norm = (xyz - xyz_mean) / xyz_std

data_norm = np.hstack([
    xyz_norm,
    cos_alpha,
    cos_beta
])

columns = ["x", "y", "z", "cos_alpha", "cos_beta"]
out_norm_df = pd.DataFrame(data_norm, columns=columns)
out_norm_df.to_csv("ROI_data_norm.csv", index=False)

pd.DataFrame(xyz_mean, columns=["x", "y", "z"]).to_csv(
    "ROI_xyz_mean.csv", index=False
)
pd.DataFrame(xyz_std, columns=["x", "y", "z"]).to_csv(
    "ROI_xyz_std.csv", index=False
)
