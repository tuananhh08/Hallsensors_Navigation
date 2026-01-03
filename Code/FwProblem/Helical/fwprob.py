import numpy as np
import pandas as pd 

#load input files
helical_df = pd.read_csv("Helical_points_coordinates.csv")
sensor_df = pd.read_csv("Hall_sensor_positions.csv")

helical_pts = helical_df.values
sensor_pos = sensor_df.values
helical_xyz = helical_pts[:,:3]

dz = 0.025
sensor_pos[:, 2] += dz

out_df = pd.DataFrame(sensor_pos, columns=["x", "y", "z"])
out_df.to_csv("Sensors_pos.csv", index=False)

# caculate B
m0 = 1
cos_alpha = 1
cos_beta = 1

mx = m0*cos_alpha*cos_beta
my = 0
mz = 0
m = np.array([mx,my,mz])

def B_formula (point,sensor, m_vec):
    r_vec = sensor - point
    r = np.linalg.norm(r_vec) +1e-12
    m_dot_r = np.dot(m,r_vec)
    
    term1 = 3*m_dot_r *r_vec / (r**5)
    term2 = m_vec / (r**3)
    B_vec = (term1 - term2)* 1e-7
    
    return B_vec

def computeB_for_all_sensors(source_point, sensor_pos,m_vec):
    B_values = []
    for sensor in sensor_pos:
        B_vec = B_formula(source_point, sensor, m_vec)
        B_mag = np.linalg.norm(B_vec)
        B_values.append(B_mag)
    
    return np.array(B_values)

m_vec = m
outputs = []
for src in helical_xyz:
    B_all = computeB_for_all_sensors(src, sensor_pos, m_vec)
    outputs.append(B_all)
    
outputs = np.array(outputs)

outputs_df = pd.DataFrame(outputs)
outputs_df.to_csv("B_output.csv",header=None, index=False)


