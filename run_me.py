import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
import os

import tkinter as tk
from tkinter import filedialog, messagebox

#-------- Definitions --------

#---------- GUI Definitions ----------
def browse_folder(entry):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry.delete(0, tk.END)
        entry.insert(0, folder_path)

def browse_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def browse_save(entry):
    save_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if save_path:
        entry.delete(0, tk.END)
        entry.insert(0, save_path)

def start_program():
	heart_folder = heart_folder_entry.get()
	save_lv = save_lv_entry.get()
	reference_vent = reference_entry.get()
	background = background_entry.get()

	# Validate inputs
	if not all([heart_folder, save_lv, reference_vent, background]):
	    messagebox.showerror("Error", "Please provide all paths!")
	    return

	#Load background mask
	background_frame = np.load(background)
	background_frame = background_frame.astype(np.int32)

	image_folder = heart_folder
	images = []
	for filename in os.listdir(image_folder):
		if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
		    img_path = os.path.join(image_folder, filename)
		    img = cv2.imread(img_path)
		    if img is not None:
		    	img = img.astype(np.int32) 
		    	images.append(img)

	#Load the reference ventricle
	ref_vent = cv2.imread(reference_vent, cv2.IMREAD_GRAYSCALE)
	ref_vent = resize_ref_vent(images[0],ref_vent)

	black_screen = np.zeros((images[0].shape[0], images[0].shape[1]))

	i = 0
	for frame in images:
		processed_frame = clean_image(frame, background_frame)
		watershed_results = run_watershed(processed_frame, background_frame)

		labels, components = find_components(watershed_results)
		components = connect_ventricle_tail(labels, components, watershed_results)

		dist_map, grad_map = create_reference_maps(frame, ref_vent)

		pred_x = 60
		pred_y = 60
		pred_area = 2327

		shape_error_list = calculate_shape_error(components, frame, ref_vent, grad_map, dist_map) 
		dist_error_list = calculate_dist_error(components, pred_x, pred_y)
		area_error_list = calculate_area_error(components, pred_area)

		shape_thr = 25
		dist_thr = 25
		area_thr = 2327

		components, shape_error_list, dist_error_list, area_error_list = remove_outliers(components, shape_error_list, dist_error_list, area_error_list, shape_thr, dist_thr, area_thr)

		shape_error_list = normalize_list(shape_error_list)
		dist_error_list = normalize_list(dist_error_list)
		area_error_list = normalize_list(dist_error_list)

		shape_error_array = np.array(shape_error_list)
		dist_error_array = np.array(dist_error_list)
		area_error_array = np.array(area_error_list)

		shape_error_weight = 0.4
		dist_error_weight = 0.4
		area_error_weight = 0.2

		total_error_array = (shape_error_weight * shape_error_array) + (dist_error_weight * dist_error_array) + (area_error_weight * area_error_array) 

		if len(total_error_array) == 0:
			left_vent = black_screen
		else:
			min_index = np.argmin(total_error_array)
			left_vent = components[min_index]

		i += 1
		save_folder = save_lv
		save_name = 'image_' + str(i) + '.jpg'
		save_path = os.path.join(save_folder, save_name)
		cv2.imwrite(save_path, left_vent)
		
	




def obtain_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    ret, frame = cap.read()
    cap.release()

    return frame

#---------- Helper Definitions ----------
def resize_ref_vent(frame, ref_vent):
	act_height = frame.shape[0]
	act_width = frame.shape[1]
	act_dim = (act_width, act_height)
	resized_ref_vent = cv2.resize(ref_vent, act_dim, interpolation=cv2.INTER_AREA)

	return resized_ref_vent

def find_ref_vent_contour(ref_vent):

	filler, binary_ref_vent = cv2.threshold(ref_vent, 250, 255, cv2.THRESH_BINARY)
	binary_ref_vent = 255 - binary_ref_vent
	ref_vent_contours, hierarchy = cv2.findContours(binary_ref_vent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	ref_vent_contour = ref_vent_contours[0]

	return ref_vent_contour

def normalize_list(p_list):

	array = np.array(p_list)
	norm_array = array / np.linalg.norm(array)
	norm_list = norm_array.tolist()

	return norm_list

def find_angle(component):
	
	ys, xs = np.where(component == 255)
	comp_x_center = xs.mean()
	comp_y_center = ys.mean()
	center = (comp_x_center, comp_y_center)

	moments = cv2.moments(component)
	mu20 = moments['mu20'] / moments['m00']
	mu02 = moments['mu02'] / moments['m00']
	mu11 = moments['mu11'] / moments['m00']

	theta = 0.5 * np.arctan2(2*mu11, mu20 - mu02)

	return theta

def find_center(component):

	ys, xs = np.where(component == 255)
	comp_x_center = int(xs.mean())
	comp_y_center = int(ys.mean())
	center = (comp_x_center, comp_y_center)

	return center

def calculate_dist_error(components, pred_x_center, pred_y_center):
		
	dist_error_list = []
	for comp in components:

		center = find_center(comp)
		act_x_center = center[0]
		act_y_center = center[1]
		delta_x = act_x_center - pred_x_center
		delta_y = act_y_center - pred_y_center
		distance = math.sqrt((delta_x ** 2) + (delta_y ** 2))
		dist_error = distance

		dist_error_list.append(dist_error)

	return dist_error_list

def calculate_area_error(components, pred_area):

	area_error_list = []
	for comp in components:
		act_area = np.sum(comp == 255)
		area_error = abs(act_area - pred_area)
		
		area_error_list.append(area_error)

	return area_error_list

def calculate_shape_error(components, frame, ref_vent, grad_map, dist_map):
	
	shape_error_list = []
	for comp in components:
		center_comp = center_component(comp)
		rotate_center_comp = rotate_component(center_comp)
		scale_rotate_center_comp = scale_component(rotate_center_comp, frame, ref_vent)
		iterations = find_obj_to_ref_iterations(frame, ref_vent, rotate_center_comp, grad_map, dist_map)
		shape_error_list.append(iterations)

	return shape_error_list

def calc_error_thr(vent_contour, ref_contour):
	vent_contour_flat = vent_contour.reshape(-1,2)
	ref_contour_flat = ref_contour.reshape(-1,2)
	tree = cKDTree(ref_contour_flat)
	min_dists, idxs = tree.query(vent_contour_flat)

	dist_sum = np.sum(min_dists)
	dist_num = len(min_dists)
	error_thr = dist_sum / dist_num

	return error_thr

#---------- Pipeline Definitions ----------
def clean_image(frame, background_frame):
	#Noise suppression with median filtering

	noise_suppress_frame = median_filter(frame ,size = 5)

	#Increasing Contrast
	background_frame = background_frame.astype('uint8')
	kernel = np.ones((3,3), np.uint8)
	background_frame = cv2.dilate(background_frame, kernel, iterations=1)
	background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

	contrast_frame = noise_suppress_frame * 2
	contrast_frame = np.clip(contrast_frame, 0, 255)
	contrast_frame = contrast_frame.astype('uint8')
	contrast_frame = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY)

	contrast_frame = contrast_frame.astype('int32')
	background_frame = background_frame.astype('int32')
	contrast_frame = contrast_frame + background_frame
	contrast_frame = np.clip(contrast_frame, 0, 255)

	blurred_frame = contrast_frame.astype(np.uint8)
	blurred_frame = cv2.GaussianBlur(blurred_frame, ksize=(3, 17), sigmaX=0.33, sigmaY=4)

	#Creating a binary mask
	binary_frame = blurred_frame.copy()	
	binary_frame = np.where(binary_frame <= 90, 255, 0)
	binary_frame = binary_frame.astype(np.uint8)

	return binary_frame

def run_watershed(processed_frame, background_frame):
	#Find the basin areas
	kernel = np.ones((3,3), np.uint8)
	basin = cv2.dilate(processed_frame, kernel, iterations=1)

	#Create the intensity map
	basin_center = cv2.distanceTransform(processed_frame, cv2.DIST_L2, 5)
	intensity_map = basin_center.copy()

	scale = 255
	intensity_map_max = np.max(intensity_map)
	intensity_map = (intensity_map / intensity_map_max) * scale
	intensity_map = (1*intensity_map) ** (1/2)
	intensity_map = scale - intensity_map
	intensity_map = intensity_map.astype(np.uint8)
	intensity_map = cv2.cvtColor(intensity_map, cv2.COLOR_GRAY2BGR)

	#Find the basin center points

	vent_diam = 20
	thr = vent_diam / 2
	thr_bas_cent, basin_center = cv2.threshold(basin_center, thr, 255, 0) ##This is probably redundant but we keep for now in case
	basin_center = basin_center.astype(np.uint8)

	#Find regions of the basin that are unlabeled
	unknown_basin = cv2.subtract(basin, basin_center)
	background_reformat = cv2.cvtColor(background_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
	thr_br, background_reformat = cv2.threshold(background_reformat, 1, 255, 0)

	unknown_basin = unknown_basin - cv2.cvtColor(background_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
	thr_ub, unknown_basin = cv2.threshold(unknown_basin, 1, 255, 0)

	#Create different markers for different regions of the image
	ret, markers = cv2.connectedComponents(basin_center)
	markers = markers + 1
	markers[unknown_basin == 255] = 0

	#Perform the watershed algorithm
	watershed_results = cv2.watershed(intensity_map, markers)

	return watershed_results

def find_components(watershed_results):
	labels = np.unique(watershed_results)
	labels = [elem for elem in labels if (elem != -1 and elem != 1)]

	components = []
	for label in labels:
		mask = np.zeros_like(watershed_results, dtype=np.uint8)
		mask[watershed_results == label] = 255  # white = current component
		components.append(mask)

	return labels, components

def create_reference_maps(frame, ref_vent):

	#Resize the reference ventricle to match the dimensions of the frame
	act_height = frame.shape[0]
	act_width =frame.shape[1]
	act_dim = (act_width, act_height)
	resized_ref_vent = cv2.resize(ref_vent, act_dim, interpolation=cv2.INTER_AREA)

	#Create the contour
	filler, binary_ref_vent = cv2.threshold(resized_ref_vent, 250, 255, cv2.THRESH_BINARY)
	binary_ref_vent = 255 - binary_ref_vent
	ref_vent_contours, hierarchy = cv2.findContours(binary_ref_vent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	ref_vent_contour = ref_vent_contours[0]

	#Draw the contour onto a background to turn it into an image
	ref_vent_contour_img = np.zeros((act_height,act_width, 1), dtype=np.uint8)
	cv2.drawContours(ref_vent_contour_img, ref_vent_contours, -1, 255, 1) 

	#Create a distance map based on the contour
	ref_vent_contour_img = cv2.bitwise_not(ref_vent_contour_img)
	dist_map = cv2.distanceTransform(ref_vent_contour_img, cv2.DIST_L2, 3)

	grad_y, grad_x = np.gradient(dist_map)
	grad_map = [grad_x, grad_y]

	return dist_map, grad_map 

def find_obj_to_ref_iterations(frame, ref_vent, comp, grad_map, dist_map):

	act_height = frame.shape[0]
	act_width =frame.shape[1]

	comp_contours, comp_hierarchy = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	comp_contour = comp_contours[0]

	ref_vent_contour = find_ref_vent_contour(ref_vent)
	ref_vent_contour_img = np.zeros((act_height,act_width, 1), dtype=np.uint8)
	cv2.drawContours(ref_vent_contour_img, ref_vent_contour, -1, 255, 1) 

	#Find the center of mass of the reference ventricle and the actual ventricle
	ref_M = cv2.moments(ref_vent_contour)
	if ref_M['m00'] != 0:
	    center_ref_x = int(ref_M['m10'] / ref_M['m00'])
	    center_ref_y = int(ref_M['m01'] / ref_M['m00'])
	else:
	    center_ref_x, center_ref_y = 0, 0

	comp_M = cv2.moments(comp_contour)
	if comp_M['m00'] != 0:
	    comp_x = int(comp_M['m10'] / comp_M['m00'])
	    comp_y = int(comp_M['m01'] / comp_M['m00'])
	else:
	    comp_x, comp_y = 0, 0

	dx = center_ref_x - comp_x
	dy = center_ref_y - comp_y 

	comp_contour += np.array([[[dx, dy]]])
	comp_contour_img = np.zeros((act_height,act_width, 1), dtype=np.uint8)
	cv2.drawContours(comp_contour_img, comp_contour, -1, 255, 1) 

	new_comp_contour = comp_contour.copy()

	#Reconfiguring the contour
	error_thr = calc_error_thr(new_comp_contour, ref_vent_contour)
	alpha = 2 # step size; adjust for stability
	num_iterations = 0 
	grad_x = grad_map[0]
	grad_y = grad_map[1]
	while error_thr > 2.5:

	    xs = new_comp_contour[:,0,0]
	    ys = new_comp_contour[:,0,1]

	    # clip to map boundaries
	    xs = np.clip(xs, 0, dist_map.shape[1]-1)
	    ys = np.clip(ys, 0, dist_map.shape[0]-1)

	    # get gradient at each point
	    dx = grad_x[ys, xs]
	    dy = grad_y[ys, xs]

	    # move points opposite to gradient (toward minima)
	    new_comp_contour[:,0,0] = new_comp_contour[:,0,0] - alpha * dx
	    new_comp_contour[:,0,1] = new_comp_contour[:,0,1] - alpha * dy

	    new_comp_contour_img = np.zeros((act_height,act_width, 1), dtype=np.uint8)
	    cv2.drawContours(new_comp_contour_img, new_comp_contour, -1, 255, 1) 

	    error_thr = calc_error_thr(new_comp_contour, ref_vent_contour)

	    num_iterations += 1

	    if num_iterations > 25:
	    	break	

	new_comp_contour_img = np.zeros((act_height,act_width, 1), dtype=np.uint8)
	cv2.drawContours(new_comp_contour_img, new_comp_contour, -1, 255, 1) 

	return num_iterations

def connect_ventricle_tail(labels, components, watershed_results):

	possible_full_left_vents = []
	for i in range(len(components)):

		component = components[i]
		label = labels[i] #Have to add +1 because labels has an outline which is negative one valued. which does not correspond to any component
		
		comps_combined = watershed_results.copy()
		comps_combined[comps_combined == label] = 0

		band_width = 7
		center = find_center(component)
		line_values = []
		
		for dx in range(-band_width, band_width + 1):
		    for t in range(max(comps_combined.shape)):
		        x = int(round(center[0] + dx))
		        y = int(round(center[1] + t))

		        if 0 <= x < comps_combined.shape[1] and 0 <= y < comps_combined.shape[0]:
		        	line_values.append(comps_combined[y, x])
		        else:
		        	break

		line_values = np.array(line_values)
		tail_label_idx = np.where(~np.isin(line_values, (-1, 0, 1)))[0]		
		if len(tail_label_idx) > 0: 
			tail_label_idx = tail_label_idx[0]
		else:
			continue
		tail_label = line_values[tail_label_idx]
		tail_component_idx = np.where(labels == tail_label)[0][0]
		tail_component = components[tail_component_idx]

		possible_full_left_vent = component + tail_component
		possible_full_left_vents.append(possible_full_left_vent)	

	extended_components = components + possible_full_left_vents

	return extended_components

def center_component(component):
	
	matrix_x_center = component.shape[1] / 2
	matrix_y_center = component.shape[0] / 2

	ys, xs = np.where(component == 255)
	comp_x_center = xs.mean()
	comp_y_center = ys.mean()
	center = (comp_x_center, comp_y_center)

	delta_x = matrix_x_center - comp_x_center
	delta_y = matrix_y_center - comp_y_center

	shift_matrix = np.float32([[1, 0, delta_x],[0, 1, delta_y]])
	shift_comp = cv2.warpAffine(component, shift_matrix, (component.shape[1], component.shape[0]), flags=cv2.INTER_NEAREST)

	return shift_comp

def rotate_component(component): 

	ys, xs = np.where(component == 255)
	comp_x_center = xs.mean()
	comp_y_center = ys.mean()
	center = (comp_x_center, comp_y_center)

	moments = cv2.moments(component)
	mu20 = moments['mu20'] / moments['m00']
	mu02 = moments['mu02'] / moments['m00']
	mu11 = moments['mu11'] / moments['m00']

	theta = 0.5 * np.arctan2(2*mu11, mu20 - mu02)
	rot_theta = -1 * ((math.pi/2) - theta)
	rot_theta_deg = np.degrees(rot_theta)

	if abs(rot_theta_deg) > (45):
		return component

	rot_matrix = cv2.getRotationMatrix2D(center, rot_theta_deg, 1.0)
	rotate_comp = cv2.warpAffine(component, rot_matrix, (component.shape[1], component.shape[0]), flags=cv2.INTER_NEAREST)

	return rotate_comp

def scale_component(component, frame, ref_vent):

	act_height = frame.shape[0]
	act_width =frame.shape[1]
	act_dim = (act_width, act_height)
	resized_ref_vent = cv2.resize(ref_vent, act_dim, interpolation=cv2.INTER_AREA)

	filler, binary_ref_vent = cv2.threshold(resized_ref_vent, 250, 255, cv2.THRESH_BINARY)
	binary_ref_vent = 255 - binary_ref_vent

	ref_vent_img_filled = binary_ref_vent.copy()

	height, width = binary_ref_vent.shape
	mask = np.zeros((height+2, width+2), np.uint8)
	seed_point = (width//2, height//2)

	cv2.floodFill(ref_vent_img_filled, mask, seed_point, 255)

	ref_area = np.sum(ref_vent_img_filled == 255)
	comp_area = np.sum(component == 255)

	scale_factor = ref_area / comp_area

	ys, xs = np.where(component == 255)
	comp_x_center = xs.mean()
	comp_y_center = ys.mean()

	x_min, x_max = xs.min(), xs.max()
	y_min, y_max = ys.min(), ys.max()
	obj_crop = component[y_min:y_max+1, x_min:x_max+1]

	# Resize
	new_h = int(obj_crop.shape[0] * scale_factor)
	new_w = int(obj_crop.shape[1] * scale_factor)
	obj_scaled = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

	# Create new matrix and place obj_scaled at the centroid
	scaled_component = np.zeros_like(component)

	# top-left corner to place scaled object
	start_x = int(comp_x_center - new_w/2)
	start_y = int(comp_y_center - new_h/2)

	H, W = scaled_component.shape

	end_y = min(start_y + new_h, H)
	end_x = min(start_x + new_w, W)

	src_y0 = max(0, -start_y)
	src_x0 = max(0, -start_x)

	dst_y0 = max(start_y, 0)
	dst_x0 = max(start_x, 0)

	obj_h = end_y - dst_y0
	obj_w = end_x - dst_x0

	# Only place if something overlaps
	if obj_h > 0 and obj_w > 0:
	    scaled_component[dst_y0:dst_y0 + obj_h,dst_x0:dst_x0 + obj_w] = obj_scaled[src_y0:src_y0 + obj_h,src_x0:src_x0 + obj_w]

	return scaled_component

def remove_outliers(components, shape_error_list, dist_error_list, area_error_list, shape_thr, dist_thr, area_thr):

	shape_error_array = np.array(shape_error_list)
	dist_error_array = np.array(dist_error_list)
	area_error_array = np.array(area_error_list)

	print('shape_thr', shape_thr)
	print('dist_thr', dist_thr)
	print('area_thr', area_thr)

	print(shape_error_array)
	print(dist_error_array)
	print(area_error_array)


	shape_outlier_indices = np.argwhere(shape_error_array > shape_thr)
	dist_outlier_indices = np.argwhere(dist_error_array > dist_thr)
	area_outlier_indices = np.argwhere(area_error_array > area_thr)

	combined_outlier_indices = np.concatenate([shape_outlier_indices, dist_outlier_indices, area_outlier_indices])
	outlier_indices = np.unique(combined_outlier_indices)

	shape_error_array = np.delete(shape_error_array, outlier_indices)
	dist_error_array = np.delete(dist_error_array, outlier_indices)
	area_error_array = np.delete(area_error_array, outlier_indices)
	components_list_no_outlier = [val for i, val in enumerate(components) if i not in outlier_indices]

	shape_error_list_no_outlier = shape_error_array.tolist()
	dist_error_list_no_outlier = dist_error_array.tolist()
	area_error_list_no_outlier = area_error_array.tolist()

	return components_list_no_outlier, shape_error_list_no_outlier, dist_error_list_no_outlier, area_error_list_no_outlier

#-------- Main --------

root = tk.Tk()
root.title("Left Ventricle Identification GUI")
root.geometry("600x600")

# Description at the top
description_text = (
    "Heart Images Folder Path: The folder that contains all of the heart images provided to you as heart_images.\n"
    "Save Path: The folder where the identified left ventricle images will be saved.\n"
    "Reference Ventricle Path: The path to the reference_ventricle image provided to you in the references folder as reference_ventricle.\n"
    "Reference Background Path: The path to the background numpy array provided to you in the references folder as reference_background."
)
tk.Label(root, text=description_text, justify="left", wraplength=550).pack(pady=10)

# Heart Images Folder Path
tk.Label(root, text="Heart Images Folder Path:").pack(pady=5)
heart_folder_entry = tk.Entry(root, width=60)
heart_folder_entry.pack(pady=5)
tk.Button(root, text="Browse", command=lambda: browse_folder(heart_folder_entry)).pack()

# Save Left Ventricle Identification Path
tk.Label(root, text="Save Folder Path:").pack(pady=5)
save_lv_entry = tk.Entry(root, width=60)
save_lv_entry.pack(pady=5)
tk.Button(root, text="Browse", command=lambda: browse_save(save_lv_entry)).pack()

# Reference Ventricle Path
tk.Label(root, text="Reference Ventricle Path:").pack(pady=5)
reference_entry = tk.Entry(root, width=60)
reference_entry.pack(pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file(reference_entry)).pack()

# Background Path
tk.Label(root, text="Reference Background Path:").pack(pady=5)
background_entry = tk.Entry(root, width=60)
background_entry.pack(pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file(background_entry)).pack()

# Go button
tk.Button(root, text="Go", bg="green", fg="black", width=25, command=start_program).pack(pady=15)

root.mainloop()
