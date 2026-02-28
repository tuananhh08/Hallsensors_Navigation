import numpy as np
import pandas as pd

#Load input data
helical_df = pd.read_csv("Helical_points_coordinates.csv")
sensor_df  = pd.read_csv("Sensors_pos.csv")

#Source points 
helical_xyz = helical_df.iloc[:, :3].to_numpy()

#Sensor positions
sensor_pos = sensor_df.iloc[:, :3].to_numpy()

N  = helical_xyz.shape[0]   
Ns = sensor_pos.shape[0]

#Magnetic moment definition
m0 = 1
cos_alpha = 1
cos_beta  = 1

m = np.array([
    m0 * cos_alpha * cos_beta,
    0.0,
    0.0
])  
#compute B
r_vec = sensor_pos[None, :, :] - helical_xyz[:, None, :]   
r = np.linalg.norm(r_vec, axis=2) + 1e-12                  
m_dot_r = np.tensordot(r_vec, m, axes=([2], [0]))          
term1 = 3 * m_dot_r[:, :, None] * r_vec / (r[:, :, None] ** 5)
term2 = m[None, None, :] / (r[:, :, None] ** 3)

B_vec = (term1 - term2) * 1e-7                              
B_all = np.linalg.norm(B_vec, axis=2)                       

#Save raw B-field
pd.DataFrame(B_all).to_csv("B_Helical.csv", header=False, index=False)
print("B_Helical saved:", B_all.shape)

# # ==============================
# # 5. Normalize B-field (same logic as before)
# # ==============================
# B_mean = B_all.mean(axis=0, keepdims=True)
# B_std  = B_all.std(axis=0, keepdims=True) + 1e-8

# B_norm = (B_all - B_mean) / B_std

# pd.DataFrame(B_norm).to_csv("B_norm.csv", header=False, index=False)
# print("B_norm saved:", B_norm.shape)

# # ==============================
# # 6. Save mean & std
# # ==============================
# pd.DataFrame(B_mean).to_csv("B_mean.csv", header=False, index=False)
# pd.DataFrame(B_std).to_csv("B_std.csv", header=False, index=False)

# print("All files saved successfully.")
# for src in helical_xyz:
#     B_all = computeB_for_all_sensors(src, sensor_pos, m_vec)
#     outputs.append(B_all)
    
# outputs = np.array(outputs)

# outputs_df = pd.DataFrame(outputs)
# outputs_df.to_csv("B_output.csv",header=None, index=False)

