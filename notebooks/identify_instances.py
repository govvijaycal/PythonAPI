import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2

COLORMAP = {
0	:	( 0, 0, 0),      # Unlabeled
1	:	( 70, 70, 70),   # Building
2	:	(190, 153, 153), # Fence
3	:	(250, 170, 160), # Other
4	:	(220, 20, 60),   # Pedestrian
5	:	(153, 153, 153), # Pole
6	:   (157, 234, 50),  # Road line
7	:	(128, 64, 128),  # Road
8	:	(244, 35, 232),  # Sidewalk
9	:	(107, 142, 35),  # Vegetation
10	:	( 0, 0, 142),    # Car
11	:	(102, 102, 156), # Wall
12 :	(220, 220, 0)   # Traffic Sign
}

def connected_component_label(
        instance_image,          # image with istance ids (modified)
        seg_image,               # segmentation label image
        depth_image,             # depth image to distinguish overlapping objects
        u,                       # seed pixel coordinate on x-axis for object <id_number>
        v,                       # seed pixel coordinate on y-axis for object <id_number>
        id_number,               # what instance label to assign to this "blob"
        depth_threshold=2.5      # max depth (m) deviation between neighboring points to be the same object
    ):
	
	points_to_expand = set()
	points_to_expand.add((u,v))

	max_u, max_v = seg_image.shape

	while len(points_to_expand) > 0:
		node = points_to_expand.pop()	
		nu, nv = node

		instance_image[nu, nv] += id_number # label this node as object <id_number>

		for neighbor_u in [nu-1, nu, nu+1]:
			for neighbor_v in [nv-1, nv, nv+1]:
				clipped_u = np.clip(neighbor_u, 0, max_u - 1)
				clipped_v = np.clip(neighbor_v, 0, max_v - 1)

				if neighbor_u == nu and neighbor_v == nv:
					continue # current node
				elif clipped_u != neighbor_u or clipped_v != neighbor_v:
					continue # out of image bounds
				elif seg_image[neighbor_u, neighbor_v] != seg_image[nu, nv]:
					continue # not the same segmentation label id
				elif instance_image[neighbor_u, neighbor_v] != 0:
					continue # visited node that is either part of another object or irrelevant
				else:
					pass # in the image and same label id, check depth next


				depth_diff = np.abs(depth_image[neighbor_u, neighbor_v] - depth_image[nu,nv])

				if depth_diff <= depth_threshold:
					points_to_expand.add((neighbor_u, neighbor_v))

def get_valid_bboxes(instance_label_image, # id-ed connected components image (modified)
	                 seg_image,            # segmentation map used to get object type
	                 seg_blob_size_dict    # dictionary mapping object type to min blob size (valid = large enough)
	                 ):
	
	bboxes = []

	num_objects = np.max(instance_label_image) # assume 0. = background and nonzero = an object
	curr_object_id = 1.

	while curr_object_id <= num_objects:
		blob_roi  = (instance_label_image == curr_object_id).astype(np.uint8)
		blob_size = np.sum(blob_roi)

		blob_seg_id = np.max(seg_image[instance_label_image == curr_object_id]) # crude way to get object type
		min_blob_size = seg_blob_size_dict[blob_seg_id]

		if blob_size < min_blob_size:
			instance_label_image[instance_label_image == curr_object_id] -= curr_object_id # remove the label for these object
			instance_label_image[instance_label_image  > curr_object_id] -= 1.             # decrement num_objects by 1
			num_objects -= 1.
		else:

			blob_inds = np.argwhere(blob_roi > 0.)
			xmin, ymin = np.min(blob_inds, axis=0)
			xmax, ymax = np.max(blob_inds, axis=0)
			bboxes.append([curr_object_id, blob_seg_id, xmin, ymin, xmax, ymax])
			curr_object_id += 1.

	return bboxes

def detect_instances_of_class(seg_image, 
	                          depth_image, 
	                          seg_object_ids = [4, 10 ,12], 
	                          min_blob_size_by_seg_id = [50, 150, 50]
	                          ):
    
    instance_label_image = np.ones(seg_image.shape) * -1 # mark irrelevant nodes as visited from the outset
    for seg_id in seg_object_ids:
    	instance_label_image[seg_image == seg_id] = 0. # mark relevant nodes to be visited

    id_number = 1

    pixs_to_visit = np.nonzero(instance_label_image + 1.)

    seg_blob_size_dict = {}
    for seg_id, blob_size in zip(seg_object_ids, min_blob_size_by_seg_id):
    	seg_blob_size_dict[seg_id] = blob_size

    for u, v in zip(pixs_to_visit[0], pixs_to_visit[1]):
            if instance_label_image[u,v] != 0:
            	pass # visited already
            elif seg_image[u,v] not in seg_object_ids:
            	instance_label_image[u,v] = -1 # not relevant but mark as visited
            else:            	
            	connected_component_label(instance_label_image, seg_image, depth_image, u, v, id_number) # relevant object, expand the connected component
            	id_number += 1

    # for convenience, set background/irrelevant pixels to 0. now.
    instance_label_image[instance_label_image == -1] = 0.    
    bboxes = get_valid_bboxes(instance_label_image, seg_image, seg_blob_size_dict)

    return instance_label_image, bboxes


def overlay_bboxes(im_rgb, bboxes):
	# Adapted from https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
	im_overlay = np.copy(im_rgb)
	for bbox in bboxes:
		obj_id, seg_id, xmin, ymin, xmax, ymax = bbox
		dx = xmax - xmin
		dy = ymax - ymin

		bbox_color = COLORMAP[seg_id]

		cv2.rectangle(im_overlay, (ymin, xmin), (ymax, xmax), bbox_color, 2)
		# could add text, see link
	cv2.putText(im_overlay, '%d Objects' % len(bboxes), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
	return im_overlay




	
