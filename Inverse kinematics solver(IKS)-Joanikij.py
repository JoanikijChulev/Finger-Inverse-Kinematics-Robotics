import math
import numpy as np

def solver(desired_coordinates, initial_guess):
    # Define the lengths of the three phalanges ("finger bones")
    length_p = 39.8
    length_i = 22.4
    length_d = 15.8
    
    # Initialize variables
    q = initial_guess.astype(float).copy()  # Joint angles (converted to float)
    error = np.array([100.0, 100.0])  # Initial error
    count = 0  # Iteration count
    
    # Iteratively solve for joint angles until error is small enough
    while np.linalg.norm(error) > 0.0000001:
        # Calculate current position based on joint angles
        x, y = calculate_position(q, length_p, length_i, length_d)
        error = desired_coordinates - np.array([x, y])  # Calculate error
        
        # Calculate the inverse of the Jacobian matrix for the method below!!!
        j_inv = calculate_jacobian_inverse(q, length_p, length_i, length_d)
        
        # Update joint angles using the inverse Jacobian matrix and error
        delta_q = np.dot(j_inv, error)
        q += delta_q
        
        # Clip joint angles to stay within the permissible bounds
        q = clip_joint_angles(q)
        
        count += 1  # Increment iteration count
    
    return q, count, error

def calculate_position(q, length_p, length_i, length_d):
    # Calculate the x and y coordinates of the fingertip based on joint angles
    x = length_p * math.cos(q[0]) + length_i * math.cos(q[0] + q[1]) + length_d * math.cos(q[0] + (5.0/3.0) * q[1])
    y = length_p * math.sin(q[0]) + length_i * math.sin(q[0] + q[1]) + length_d * math.sin(q[0] + (5.0/3.0) * q[1])
    return x, y

def calculate_jacobian_inverse(q, length_p, length_i, length_d):
    # Calculate the 2x2 Jacobian matrix and its inverse based on joint angles
    jacobian = np.array([
        [- length_p * math.sin(q[0]) - length_i * math.sin(q[0] + q[1]) - length_d * math.sin(q[0] + (5.0/3.0) * q[1]), 
         - length_i * math.sin(q[0] + q[1]) - (5.0/3.0) * length_d * math.sin(q[0] + (5.0/3.0) * q[1])],
        [length_p * math.cos(q[0]) + length_i * math.cos(q[0] + q[1]) + length_d * math.cos(q[0] + (5.0/3.0) * q[1]), 
         length_i * math.cos(q[0] + q[1]) + (5.0/3.0) * length_d * math.cos(q[0] + (5.0/3.0) * q[1])]
    ])
    
    j_inv = np.linalg.inv(jacobian)  # Calculate the inverse of the Jacobian matrix (quite literally)
    return j_inv

def clip_joint_angles(q):
    # Clip joint angles to stay within the permissible bounds
    q[0] = np.clip(q[0], -math.pi / 3.0, math.pi / 3.0)
    q[1] = np.clip(q[1], -(2.0 * math.pi) / 3.0, 0)
    return q

def main():
    # Desired final coordinates and initial guess for joint angles
    desired_coordinates = np.array([40, -18])
    initial_guess = np.array([0, -2])
    
    # Solve for joint angles, get iteration count and error
    joint_angles, iterations, error = solver(desired_coordinates, initial_guess)
    
    # Print the results
    print("Joint angles for the metacarpophalangeal (MCP) and proximal interphalangeal (PIP) joints are:", joint_angles)
    print("Number of iterations it took:", iterations)
    print("Errors for the estimates for x and y are:", error)
    print("Final coordinates x,y of the fingertip are:", calculate_position(joint_angles, 39.8, 22.4, 15.8))

if __name__ == "__main__":
    main()
