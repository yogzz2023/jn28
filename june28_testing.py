import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

r=[]
el=[]
az=[]

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)
        
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("pp:", self.Pp)

    def update_step(self, Z):  # Update step with JPDA
        Inn = Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def gating(self, Z):
        # Calculate Mahalanobis distance to check if measurement Z is within gate
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def generate_hypotheses(clusters, targets):
    # Generate all possible hypotheses
    hypotheses = []
    for cluster in clusters:
        for target in targets:
            hypotheses.append((cluster, target))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster, target = hypothesis
    Z = np.array([[cluster[0]], [cluster[1]], [cluster[2]]])
    Inn = Z - np.dot(filter_instance.H, target)
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(measurements, targets, filter_instance):
    clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z).item():
            clusters.append(measurement)
    hypotheses = generate_hypotheses(clusters, targets)
    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, filter_instance) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)
    
    # Handle division by zero
    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis = hypotheses[np.argmax(marginal_probabilities)]
    return best_hypothesis

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)    

    if x > 0.0:                
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az       
        
    az = az * 180 / 3.14 

    if az < 0.0:
        az = 360 + az
    
    if az > 360:
        az = az - 360   
      
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'test.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Initial targets list
targets = []

# Iterate through measurements
for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
        targets.append(kalman_filter.Sf)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_r, prev_az, prev_el = measurements[i-1][:3]
        dt = mt - measurements[i-1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
        targets.append(kalman_filter.Sf)
    else:
        # Perform JPDA for associating measurements
        best_hypothesis = jpda(measurements[:i], targets, kalman_filter)
        Z = np.array([[best_hypothesis[0][0]], [best_hypothesis[0][1]], [best_hypothesis[0][2]]])
        
        kalman_filter.predict_step(mt)
        kalman_filter.update_step(Z)

        # Append data for plotting
        time_list.append(mt)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[0], label='filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
