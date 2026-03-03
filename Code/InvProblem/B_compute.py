import numpy as np
import pandas as pd

#load input data
helical_df = pd.read_csv("Helical_points_coordinates.csv")
sensor_df = pd.read_csv("Sensors_pos.csv")
gendata_df = pd.read_csv("ROI_data.csv")

helical_pts = helical_df.values
helical_xyz = helical_pts[:,:3]
sensor_pos = sensor_df.values
roi_points = xyz_df = gendata_df.iloc[:, :3].to_numpy()

N = roi_points.shape[0]
Ns = sensor_pos.shape[0]

m0 = 1
cos_alpha = 1
cos_beta = 1

mx = m0*cos_alpha*cos_beta
my = 0
mz = 0
m = np.array([mx,my,mz])  

def B_formula (point,sensor, m):
    
    r_vec = sensor - point
    r = np.linalg.norm(r_vec) +1e-12
    
    m_dot_r = np.dot(m,r_vec)
    
    term1 = 3*m_dot_r *r_vec / (r**5)
    term2 = m / (r**3)
    
    B_vec = (term1 - term2)* 1e-7
    B = np.linalg.norm(B_vec)
    
    return B

#compute B 
B_all = np.zeros((N, Ns))

for i in range(N):
    for j in range(Ns):
        B_all[i, j] = B_formula(
            roi_points[i],
            sensor_pos[j],
            m
        )
B_all_df = pd.DataFrame(B_all)
B_all_df.to_csv("B_all.csv",header=False,index=False)
print("B-field computed:", B_all.shape)

#normalize B-field
B_mean = B_all.mean(axis=0, keepdims=True)
B_std = B_all.std(axis=0, keepdims=True) +1e-8

B_norm = (B_all - B_mean) / B_std

B_norm_df = pd.DataFrame(B_norm)
B_norm_df.to_csv("B_norm.csv", header = False, index = False)
print("B-field normalized", B_norm.shape)

print("files saved")

#save mean &std
pd.DataFrame(B_mean).to_csv('B_mean.csv', header=False, index=False)
pd.DataFrame(B_std).to_csv('B_std.csv', header=False, index=False)
