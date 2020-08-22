import carla
import pygame
import random
import argparse
import cv2
import numpy as np
import pdb

from synchronous_mode import CarlaSyncMode, draw_image, get_font, should_quit

def create_camera_from_config(world, vehicle, bp_library, camera_config):
    bp = bp_library.find('sensor.camera.rgb')

    bp.set_attribute('image_size_x', str(camera_config['width']))
    bp.set_attribute('image_size_y', str(camera_config['height']))
    bp.set_attribute('fov', str(camera_config['fov']))

    sensor_location = carla.Location(x=camera_config['x'], y=camera_config['y'],
                                     z=camera_config['z'])
    sensor_rotation = carla.Rotation(pitch=camera_config['pitch'],
                                     roll=camera_config['roll'],
                                     yaw=camera_config['yaw'])
    sensor_transform = carla.Transform(sensor_location, sensor_rotation)

    return world.spawn_actor(bp, sensor_transform, attach_to=vehicle)



def main_autopilot(args, max_frames=1000):
    actor_list = []
    pygame.init()

    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        bp_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(bp_library.filter(args.filter)),
            start_pose)
        vehicle.set_autopilot(True)
        ego_id = vehicle.id
        actor_list.append(vehicle)


        # Spectator
        spectator = world.get_spectator()

        # Center, Left, # Right
        camera_configs = [ \
         {'x':0.7, 'y':0.0, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
         'width':960, 'height': 720, 'fov':120},
         {'x':-0.7, 'y':-0.4, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':225.0, \
         'width':960, 'height': 720, 'fov':120},
         {'x':-0.7, 'y':0.4, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':135.0, \
         'width':960, 'height': 720, 'fov':120},
         {'x':0.7, 'y':0.0, 'z':1.6, \
         'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
         'width':960, 'height': 720, 'fov':60},
        ]

        bp_cam_center = bp_library.find('sensor.camera.rgb')
        bp_cam_left   = bp_library.find('sensor.camera.rgb')
        bp_cam_right  = bp_library.find('sensor.camera.rgb')
        bp_cam_far    = bp_library.find('sensor.camera.rgb')

        cam_center = create_camera_from_config(world, vehicle, bp_library, camera_configs[0])
        cam_left   = create_camera_from_config(world, vehicle, bp_library, camera_configs[1])
        cam_right  = create_camera_from_config(world, vehicle, bp_library, camera_configs[2])
        cam_far  = create_camera_from_config(world, vehicle, bp_library, camera_configs[3])

        actor_list.append(cam_center)
        actor_list.append(cam_left)
        actor_list.append(cam_right)
        actor_list.append(cam_far)

        # Create a synchronous mode context.
        num_frames_saved = 0
        with CarlaSyncMode(world, cam_center, cam_left, cam_right, cam_far, fps=args.fps) as sync_mode:
            while True:
                if should_quit() or num_frames_saved >= max_frames:
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_center, image_left, image_right, image_far = sync_mode.tick(timeout=2.0)
                pdb.set_trace()

                ego_snap = snapshot.find(ego_id)
                vel_ego = ego_snap.get_velocity()
                vel_thresh = 1.0
                if vel_ego.x**2 + vel_ego.y**2 > vel_thresh:
                    # image_rgb.save_to_disk('%s/rgb/%08d' % (args.logdir, num_frames_saved))
                    # image_depth.save_to_disk('%s/depth/%08d' % (args.logdir, num_frames_saved))
                    # image_semseg.save_to_disk('%s/seg/%08d' % (args.logdir, num_frames_saved))
                    num_frames_saved +=1
                    print('Frames Saved: %d of %d' % (num_frames_saved, max_frames))

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                img_array = np.zeros((360, 1440, 3), dtype=np.uint8)

                center_array = np.frombuffer(image_center.raw_data, dtype=np.uint8)
                center_array = np.reshape(center_array, (image_center.height, image_center.width, 4))
                center_array = center_array[:, :, :3]
                center_array = cv2.resize(center_array, (480, 360))

                left_array = np.frombuffer(image_left.raw_data, dtype=np.uint8)
                left_array = np.reshape(left_array, (image_left.height, image_left.width, 4))
                left_array = left_array[:, :, :3]
                left_array = cv2.resize(left_array, (480, 360))

                right_array = np.frombuffer(image_right.raw_data, dtype=np.uint8)
                right_array = np.reshape(right_array, (image_right.height, image_right.width, 4))
                right_array = right_array[:, :, :3]
                right_array = cv2.resize(right_array, (480, 360))

                far_array = np.frombuffer(image_far.raw_data, dtype=np.uint8)
                far_array = np.reshape(far_array, (image_far.height, image_far.width, 4))
                far_array = far_array[:, :, :3]
                far_array = cv2.resize(far_array, (480, 360))

                img_array[:, :480, :]    = left_array
                img_array[:, 480:960, :] = center_array
                img_array[:, 960:, :]    = right_array 
                cv2.imshow('fused', img_array);
                cv2.imshow('far', far_array)
                cv2.waitKey(1)

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        cv2.destroyAllWindows()

        pygame.quit()
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