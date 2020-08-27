
def extract_vector_3D(carla_vec_3D):
	return {'x': carla_vec_3D.x,
	        'y': carla_vec_3D.y,
	        'z': carla_vec_3D.z}

def extract_rotation_3D(carla_rot_3D):
	# see this, confused about Euler angle order.  I think its pitch, yaw, roll (YZX).
	# https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation 
	return {'roll':  carla_rot_3D.roll, 
	        'pitch': carla_rot_3D.pitch, 
	        'yaw':   carla_rot_3D.yaw}

def extract_transform(carla_transform):
	return extract_vector_3D(carla_transform.location), \
	       extract_rotation_3D(carla_transform.rotation)

def extract_bbox(carla_bbox, actor_transform):
	bbox_dict = {}
	bbox_dict['extent'] = extract_vector_3D(carla_bbox.extent)
	bbox_dict['location'] = extract_vector_3D(carla_bbox.location)
	bbox_dict['local_vertices'] = [extract_vector_3D(vertex) for vertex in carla_bbox.get_local_vertices()]
	bbox_dict['world_vertices'] = [extract_vector_3D(vertex) for vertex in carla_bbox.get_world_vertices(actor_transform)]
	return bbox_dict

def extract_control(carla_control, control_type='vehicle'):
	control_dict = {}
	if control_type == 'vehicle':
		attrs = ['throttle', 'steer', 'brake', 'hand_brake', 'reverse', 'manual_gear_shift', 'gear']
	elif control_type == 'walker':
		attrs = ['direction', 'speed', 'jump']
	else:
		raise ValueError("Invalid control_type: %s" % control_type)

	for attr in attrs:
		if attr == 'direction':
			control_dict['direction'] = extract_vector_3D(carla_control.direction)
		else:
			control_dict[attr] = getattr(carla_control, attr)
	return control_dict

def process_world_snapshot(snapshot, world):
	snapshot_dict = {}

	# World Snapshot Information
	for attr in ['id', 'frame', 'delta_seconds', 'elapsed_seconds']:
		snapshot_dict[attr] = getattr(snapshot, attr)

	for actor_snap in snapshot:
		actor_dict = {}

		actor_dict['id'] =  actor_snap.id
		actor_dict['velocity']         = extract_vector_3D( actor_snap.get_velocity() )
		actor_dict['angular_velocity'] = extract_vector_3D( actor_snap.get_angular_velocity() )
		actor_dict['acceleration']     = extract_vector_3D( actor_snap.get_acceleration() )
		actor_dict['location'], actor_dict['rotation'] = extract_transform( actor_snap.get_transform() )

		actor = world.get_actor(actor_dict['id'])
		actor_dict['type_id'] = actor.type_id

		if actor.parent is not None:
			actor_dict['parent'] = actor.parent.id

		for (k, v) in actor.attributes.items():
			actor_dict[k] = v

		# Specialized Processing By Actor Type.
		if 'spectator' in actor.type_id or \
		   'traffic.unknown' in actor.type_id or \
		   'controller.ai.walker' in actor.type_id or \
		   'static.prop' in actor.type_id:
			# We choose to ignore these actors since they are irrelevant for scene reconstruction.
			# These are NOT logged to the final dictionary.
			continue
		elif 'traffic.stop' in actor.type_id or \
		     'traffic.speed_limit' in actor.type_id or \
		     'traffic.yield' in actor.type_id:
			process_traffic_sign_actor(actor, actor_dict)
		elif 'traffic.traffic_light' in actor.type_id:
			process_traffic_light_actor(actor, actor_dict)
		elif 'vehicle' in actor.type_id:
			process_vehicle_actor(actor, actor_dict)
		elif 'walker' in actor.type_id:
			process_walker_actor(actor, actor_dict)
		elif 'sensor.camera' in actor.type_id:
			process_camera_actor(actor, actor_dict)
		else:
			raise ValueError('Did not expect this actor type: %s' % actor.type_id)


		# Act actor dictionary using id as key to world snapshot dict.
		actor_key = actor_dict['type_id'] + '_' + str(actor_dict['id'])
		snapshot_dict[ actor_key ] = actor_dict

	return snapshot_dict

def process_traffic_sign_actor(actor, actor_dict):
	actor_dict['bounding_box'] = extract_bbox(actor.trigger_volume, actor.get_transform())

def process_traffic_light_actor(actor, actor_dict):
	actor_dict['bounding_box'] = extract_bbox(actor.trigger_volume, actor.get_transform())
	actor_dict['light_state']  = str(actor.state)
	actor_dict['pole_index']   = actor.get_pole_index()

def process_vehicle_actor(actor, actor_dict):
	actor_dict['bounding_box'] = extract_bbox(actor.bounding_box, actor.get_transform())
	actor_dict['speed_limit'] = actor.get_speed_limit()
	actor_dict['control'] = extract_control(actor.get_control(), control_type='vehicle')
	actor_dict['light_state'] = str(actor.get_light_state())

	if actor.get_traffic_light():
		actor_dict['traffic_light_id'] = actor.get_traffic_light().id
		actor_dict['traffic_light_state'] = str(actor.get_traffic_light_state())

def process_walker_actor(actor, actor_dict):
	actor_dict['bounding_box'] = extract_bbox(actor.bounding_box, actor.get_transform())
	actor_dict['control'] = extract_control(actor.get_control(), control_type='walker')

def process_camera_actor(actor, actor_dict):
	pass # Don't think we need anything else.  Most settings are in attributes.
