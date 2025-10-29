import numpy as np
import matplotlib.pyplot as plt
import scipy

##########################################
##########################################

#Has to be switched to False manually if to be used for fitting in amplitudes
def Arbitrary_Gauss_function(x_y, x0, y0, I0, sx, sy, angle, offset):
    x, y = x_y
    
    position_array = np.expand_dims(np.moveaxis(np.array([x - x0, y - y0]),0,-1),-2)
    
    sigma_matrix = np.array([[sx**2, 0], 
                             [0, sy**2]])
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], 
                                [-np.sin(angle), np.cos(angle)]])
    rotated_sigma_matrix = rotation_matrix @ sigma_matrix @ rotation_matrix.transpose()
    
    Gauss_amplitude = np.exp(-1/2 * position_array @ np.linalg.inv(rotated_sigma_matrix) @ np.moveaxis(position_array, -2,-1))
    
    Gauss_intensity = Gauss_amplitude**2
    
    return I0 * np.squeeze(Gauss_intensity) + offset

def Fit_Arbitrary_Gauss(data, radius_est, angle_est):
    X_Y_grids = np.indices((data.shape[0],data.shape[1]))                  #Generate XY grids of (2,p,p) shape with top_left_corner=(0,0), i.e., Airy center is app. at (400,400)
    X, Y = X_Y_grids                                                       #Separate x and y grids  to (p,p) and (p,p)
    xy_for_fit = np.vstack((X.ravel(), Y.ravel()))                         #Flatten and stack to obtain (2, p**2) arranged correctly
    z_for_fit = data.ravel()                                               #Flatten the intensity grid of chosen sample to (p**2)
    
    x0_est, y0_est = np.unravel_index(np.argmax(data), data.shape)
    I0_est = data.max()
    offset_est = data.min()
    
    p_0 = (x0_est, y0_est, I0_est, radius_est, radius_est, angle_est, offset_est)   #Initial guess
    p_optimal, p_covariance = scipy.optimize.curve_fit(Arbitrary_Gauss_function, xy_for_fit, z_for_fit, p0=p_0, 
                                                       bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf], 
                                                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf]))
    p_stds = np.sqrt(np.diag(p_covariance))
    return p_optimal, p_stds

def Find_center(data, restriction, radius_est = 5, angle_est = np.pi/2, fit_size = 5, plot=False, return_all_params=False):
    a1, a2, b1, b2 = restriction
    
    data_cut = data[a1:a2,b1:b2]
    if plot:
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.matshow(data_cut, cmap="cividis", fignum=False)
        plt.title("Data temporal difference")
    
    max_pos_1, max_pos_2 = np.unravel_index(np.argmax(data_cut), data_cut.shape)
    max_pos_1 += a1
    max_pos_2 += b1
    
    #Restriction the area for fit to position of maximum +- fit_size_, unless it would cross the image border
    fit_range_a1 = int(np.max([0, max_pos_1-fit_size]))
    fit_range_a2 = int(np.min([max_pos_1+fit_size+1, data.shape[0]]))
    fit_range_b1 = int(np.max([0, max_pos_2-fit_size]))
    fit_range_b2 = int(np.min([max_pos_2+fit_size+1, data.shape[1]]))
    
    data_fit = data[fit_range_a1:fit_range_a2, fit_range_b1:fit_range_b2] - data.min()
    
    (x, y, I, sigma_x, sigma_y, angle, offset), p_stds = Fit_Arbitrary_Gauss(data_fit, radius_est, angle_est)
    x_pos = x + max_pos_1 - fit_size
    y_pos = y + max_pos_2 - fit_size

    if plot:
        plt.subplot(132)
        plt.matshow(data_fit, cmap="cividis", fignum=False)
        plt.title("Data to fit")
        plt.subplot(133)
        X_Y_grids = np.indices((2*fit_size+1, 2*fit_size+1))
        plt.matshow(Arbitrary_Gauss_function(X_Y_grids, x, y, I, sigma_x, sigma_y, angle, offset), cmap="cividis", fignum=False)
        plt.title("Fitted")
        plt.show()
    if return_all_params:
        return (x_pos, y_pos, I, sigma_x, sigma_y, angle, offset), p_stds
    else:
        return (x_pos, y_pos), (p_stds[0], p_stds[1])

def Find_given_centers(image, restr_list, plot=False, fit_size=8, inner_plot=False):
    if plot:
        plt.matshow(image)
    
    pos_list = np.zeros([len(restr_list), 2])
    std_list = np.zeros([len(restr_list), 2])
    sigmas_list = np.zeros([len(restr_list), 2])
    for i in range(len(restr_list)):
        current_parameters, current_stds = Find_center(image, restr_list[i], fit_size=fit_size, plot=inner_plot, return_all_params=True)
        current_x, current_y, _, current_sigma_x, current_sigma_y, _, _ = current_parameters
        current_x_std, current_y_std, _, _, _, _, _ = current_stds
        
        pos_list[i,:] = current_x, current_y
        std_list[i,:] = current_x_std, current_y_std
        sigmas_list[i,:] = current_sigma_x, current_sigma_y
        
        if plot:
            plt.plot(current_y, current_x, '+', color="red")
    
    return pos_list, std_list, sigmas_list

##########################################

def Gaussian_sigma_to_Airy_min(sigma):
    Gauss_HWHM = np.sqrt(np.log(2)) * sigma
    
    Airy_HWHM_table_value = 1.6163
    Airy_1min_table_value = 3.8317
    Airy_HWHM_to_1min_factor = Airy_1min_table_value / Airy_HWHM_table_value
    
    Airy_HWHM = Gauss_HWHM
    Airy_1min = Airy_HWHM * Airy_HWHM_to_1min_factor
    
    return Airy_1min

##########################################

def Double_Arbitrary_Gauss_function(x_y, x0_1, y0_1, I0_1, sx_1, sy_1, angle_1, 
                                    x0_2, y0_2, I0_2, sx_2, sy_2, angle_2, offset):
    x, y = x_y
    
    position_array_1 = np.expand_dims(np.moveaxis(np.array([x - x0_1, y - y0_1]),0,-1),-2)
    sigma_matrix_1 = np.array([[sx_1**2, 0], [0, sy_1**2]])
    rotation_matrix_1 = np.array([[np.cos(angle_1), np.sin(angle_1)], 
                                  [-np.sin(angle_1), np.cos(angle_1)]])
    rotated_sigma_matrix_1 = rotation_matrix_1 @ sigma_matrix_1 @ rotation_matrix_1.transpose()
    
    position_array_2 = np.expand_dims(np.moveaxis(np.array([x - x0_2, y - y0_2]),0,-1),-2)
    sigma_matrix_2 = np.array([[sx_2**2, 0], [0, sy_2**2]])
    rotation_matrix_2 = np.array([[np.cos(angle_2), np.sin(angle_2)], 
                                  [-np.sin(angle_2), np.cos(angle_2)]])
    rotated_sigma_matrix_2 = rotation_matrix_2 @ sigma_matrix_2 @ rotation_matrix_2.transpose()
    
    Gauss_amplitude_1 = np.exp(-1/2 * position_array_1 @ np.linalg.inv(rotated_sigma_matrix_1) @ np.moveaxis(position_array_1, -2,-1))
    Gauss_amplitude_2 = np.exp(-1/2 * position_array_2 @ np.linalg.inv(rotated_sigma_matrix_2) @ np.moveaxis(position_array_2, -2,-1))
    
    Gauss_intensity_1 = I0_1 * Gauss_amplitude_1**2
    Gauss_intensity_2 = I0_2 * Gauss_amplitude_2**2
    
    return np.squeeze(Gauss_intensity_1) + np.squeeze(Gauss_intensity_2) + offset

def Fit_Double_Arbitrary_Gauss(data, radius_est, 
                               x0_1_est=9, x0_2_est=9, y0_1_est=3, y0_2_est=8):
    X_Y_grids = np.indices((data.shape[0],data.shape[1]))                  #Generate XY grids of (2,p,p) shape with top_left_corner=(0,0), i.e., Airy center is app. at (400,400)
    X, Y = X_Y_grids                                                       #Separate x and y grids  to (p,p) and (p,p)
    xy_for_fit = np.vstack((X.ravel(), Y.ravel()))                         #Flatten and stack to obtain (2, p**2) arranged correctly
    z_for_fit = data.ravel()                                               #Flatten the intensity grid of chosen sample to (p**2)
    
    I0_est = data.max()
    offset_est = data.min()
    angle_est = 0
    
    p_0 = (x0_1_est, y0_1_est, I0_est, radius_est, radius_est, angle_est, 
           x0_2_est, y0_2_est, I0_est, radius_est, radius_est, angle_est, 
           offset_est)   #Initial guess
    p_optimal, p_covariance = scipy.optimize.curve_fit(Double_Arbitrary_Gauss_function, xy_for_fit, z_for_fit, p0=p_0, 
                                                       bounds=([0, 0, 0, 0, 0, 0, 
                                                                0, 0, 0, 0, 0, 0, 0], 
                                                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, 
                                                                np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf]))
    
    p_stds = np.sqrt(np.diag(p_covariance))
    return p_optimal, p_stds









