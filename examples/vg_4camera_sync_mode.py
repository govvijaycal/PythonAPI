import carla
import os
import sys
import random
import argparse
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import json
import threading
import time

import copy
from queue import Queue
from synchronous_mode import CarlaSyncMode
from vg_snapshot_utils import process_world_snapshot

images_queue = Queue() #[carla.Image, saveloc]

def image_saving_thread_function():
    while True:
        if images_queue.qsize() > 0:            
            img, saveloc = images_queue.get()
            img.save_to_disk(saveloc)
            images_queue.task_done()
        else:
            time.sleep(0.001)

def create_cameras_from_config(world, vehicle, bp_library, camera_config):
    sensor_location = carla.Location(x=camera_config['x'], y=camera_config['y'],
                                     z=camera_config['z'])
    sensor_rotation = carla.Rotation(pitch=camera_config['pitch'],
                                     roll=camera_config['roll'],
                                     yaw=camera_config['yaw'])
    sensor_transform = carla.Transform(sensor_location, sensor_rotation)

    bp_rgb = bp_library.find('sensor.camera.rgb')
    bp_seg = bp_library.find('sensor.camera.semantic_segmentation')
    bp_depth = bp_library.find('sensor.camera.depth')

    for bp in [bp_rgb, bp_seg, bp_depth]:
        bp.set_attribute('image_size_x', str(camera_config['width']))
        bp.set_attribute('image_size_y', str(camera_config['height']))
        bp.set_attribute('fov', str(camera_config['fov']))
        bp.set_attribute('role_name', str(camera_config['name']))

    return [world.spawn_actor(bp, sensor_transform, attach_to=vehicle) for bp in [bp_rgb, bp_seg, bp_depth]]

def main_autopilot(args, max_frames=50):
    print('Hit the Escape Key on the OpenCV window to exit.')
    actor_list  = []
    camera_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    ret = 0

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        bp_library = world.get_blueprint_library()
        bp_vehicle = random.choice(bp_library.filter(args.filter))
        bp_vehicle.set_attribute('role_name', 'hero')

        vehicle = world.spawn_actor(bp_vehicle, start_pose)
        vehicle.set_autopilot(True)
        ego_id = vehicle.id
        actor_list.append(vehicle)

        # Camera parameter settings.
        camera_configs = [ \
         {'x':0.7, 'y':0.0, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
         'width':960, 'height': 720, 'fov':120, \
         'name':'center'},
         {'x':-0.7, 'y':-0.4, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':225.0, \
         'width':960, 'height': 720, 'fov':120, \
         'name': 'left'},
         {'x':-0.7, 'y':0.4, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':135.0, \
         'width':960, 'height': 720, 'fov':120, \
         'name': 'right'},
         {'x':0.7, 'y':0.0, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
         'width':960, 'height': 720, 'fov':60, \
         'name': 'far'},
        ]

        # Create all cameras and add to actor/camera lists.  Also create folders to save to.
        rgb_center, seg_center, depth_center = create_cameras_from_config(world, vehicle, bp_library, camera_configs[0])
        rgb_left,   seg_left,   depth_left   = create_cameras_from_config(world, vehicle, bp_library, camera_configs[1])
        rgb_right,  seg_right,  depth_right  = create_cameras_from_config(world, vehicle, bp_library, camera_configs[2])
        rgb_far,    seg_far,    depth_far    = create_cameras_from_config(world, vehicle, bp_library, camera_configs[3])
        
        for camera_id in ['center', 'left', 'right', 'far']:
            for camera_type in ['rgb', 'seg', 'depth']:

                camera_name = camera_type + '_' + camera_id            
                camera_list.append(locals()[camera_name])
                os.makedirs('%s/%s/' % (args.logdir, camera_name), exist_ok=True)

        actor_list.extend(camera_list)
        os.makedirs('%s/snapshot/' % args.logdir, exist_ok=True)        

        # Start Image Saving Thread.  Probably smarter way of doing this, but threads = #image producers works okay.
        [threading.Thread(target=image_saving_thread_function, daemon=True).start() for x in range(12)]

        # Create a synchronous mode context.
        num_frames_saved = 0
        with CarlaSyncMode(world, *camera_list, fps=args.fps) as sync_mode:
            while True:
                # Advance the simulation and wait for the data.
                snapshot, \
                image_rgb_center, image_seg_center, image_depth_center, \
                image_rgb_left,   image_seg_left,   image_depth_left, \
                image_rgb_right,  image_seg_right,  image_depth_right, \
                image_rgb_far,    image_seg_far,    image_depth_far = sync_mode.tick(timeout=2.0)

                ego_snap = snapshot.find(ego_id)
                vel_ego = ego_snap.get_velocity()
                vel_thresh = 1.0
                if vel_ego.x**2 + vel_ego.y**2 < vel_thresh:
                    print('Waiting for ego to move.', file=sys.stderr)
                else:

                    # Write world snapshot to json.
                    snapshot_dict = process_world_snapshot(snapshot, world)
                    json_name = '%s/snapshot/%08d.json' % (args.logdir, num_frames_saved)
                    with open(json_name, 'w') as f:
                        f.write(json.dumps(snapshot_dict, indent=4))

                    # Process collecting images.  Visualization for debugging.
                    mosaic_array = np.zeros((540, 960, 3), dtype=np.uint8)
                    for col_idx, camera_id in  enumerate(['center', 'left', 'right', 'far']):
                        for row_idx, camera_type in enumerate(['rgb', 'seg', 'depth']):
                            img_name = 'image_' + camera_type + '_' + camera_id
                            img = locals()[img_name]

                            savedir = args.logdir + '/' + camera_type + '_' + camera_id + '/'
                            # img.save_to_disk('%s/%08d' % (savedir, num_frames_saved))                            
                            images_queue.put([img, '%s/%08d' % (savedir, num_frames_saved)])                            

                            """
                            # Issue: converting the img adjust img in the queue too.
                            # Image is not "deepcopy"-able as well.  
                            # So just view or save one at a time for now.
                            if camera_type == 'seg':
                                img.convert(carla.ColorConverter.CityScapesPalette)
                            elif camera_type == 'depth':
                                img.convert(carla.ColorConverter.LogarithmicDepth)

                            img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
                            img_array = np.reshape(img_array, (img.height, img.width, 4))
                            img_array = img_array[:, :, :3]

                            if camera_type == 'rgb':
                                img_array = cv2.resize(img_array, (240, 180), cv2.INTER_CUBIC)
                            elif camera_type == 'seg' or camera_type == 'depth':
                                img_array = cv2.resize(img_array, (240, 180), cv2.INTER_NEAREST)
                            else:
                                raise ValueError("Unhandled image type: %s" % camera_type)

                            xmin = col_idx * 240
                            xmax = (col_idx + 1) * 240
                            ymin = row_idx * 180
                            ymax = (row_idx + 1) * 180

                            mosaic_array[ymin:ymax, xmin:xmax, :] = img_array
                            """
                    
                    cv2.imshow('mosaic', mosaic_array)
                    ret = cv2.waitKey(10)

                    num_frames_saved +=1
                    fps = round(1.0 / snapshot.timestamp.delta_seconds)
                    print('Frames Saved: %d of %d, fps: %.1f, QSize: %d' % (num_frames_saved, max_frames, fps, images_queue.qsize()), file=sys.stderr)
                    time.sleep(0.1)

                if num_frames_saved >= max_frames or ret == 27: # Esc
                    return                

    except Exception as e:
        print(e)

    finally:
        print('Waiting for all images to be saved.')
        images_queue.join()        

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        cv2.destroyAllWindows()

        print('done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Synchronous Camera Data Collector')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln.*',
        help='actor filter (default: "vehicle.lincoln.*")')
    argparser.add_argument( 
        '--logdir',
        default='data_synced',
        help='Image logging directory for saved rgb,depth,and semantic segmentation images.')
    argparser.add_argument( 
        '--fps',
        default=5,
        type=int)
    args = argparser.parse_args()

    try:
        main_autopilot(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')