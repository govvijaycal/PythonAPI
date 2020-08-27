import carla
import pygame
import random
import argparse
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import json

from synchronous_mode import CarlaSyncMode, draw_image, get_font, should_quit
from vg_snapshot_utils import process_world_snapshot


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

    return [world.spawn_actor(bp, sensor_transform, attach_to=vehicle) for bp in [bp_rgb, bp_seg, bp_depth]]



def main_autopilot(args, max_frames=1000):
    actor_list = []
    camera_list = []
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

        cam_center, seg_center, depth_center = create_cameras_from_config(world, vehicle, bp_library, camera_configs[0])
        cam_left, seg_left, depth_left       = create_cameras_from_config(world, vehicle, bp_library, camera_configs[1])
        cam_right, seg_right, depth_right    = create_cameras_from_config(world, vehicle, bp_library, camera_configs[2])
        cam_far, seg_far, depth_far          = create_cameras_from_config(world, vehicle, bp_library, camera_configs[3])
        

        for camera_id in ['center', 'left', 'right', 'far']:
            for camera_type in ['cam', 'seg', 'depth']:

                camera_name = camera_type + '_' + camera_id
                camera_list.append(locals()[camera_name])

        actor_list.extend(camera_list)
        
        # Create a synchronous mode context.
        num_frames_saved = 0
        with CarlaSyncMode(world, *camera_list, fps=args.fps) as sync_mode:
            while True:
                if should_quit() or num_frames_saved >= max_frames:
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, \
                image_center, image_center_seg, image_center_depth, \
                image_left, image_left_seg, image_left_depth, \
                image_right, image_right_seg, image_right_depth, \
                image_far, image_far_seg, image_far_depth = sync_mode.tick(timeout=2.0)

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

                mosaic_array = np.zeros((1080, 1920, 3), dtype=np.uint8)
                
                # actors = world.get_actors()

                # waypoint_tuple_list = world.get_map().get_topology()
                # for (pt1, pt2) in waypoint_tuple_list:
                #     start = [pt1.transform.location.x, pt1.transform.location.y]
                #     end   = [pt2.transform.location.x, pt2.transform.location.y]
                #     plt.plot(start[0], start[1], 'rx')
                #     plt.plot(end[0], end[1], 'bx')
                #     plt.plot([start[0], end[0]], [start[1], end[1]], 'k')

                # # for actor in actors:
                #     loc = actor.get_location()

                #     if actor.type_id == 'spectator':
                #         pass
                #     if actor.type_id == 'vehicle':
                #         plt.plot(loc.x, loc.y, 'c*')
                #     elif actor.type_id == 'traffic.unknown':
                #         plt.plot(loc.x, loc.y, 'm+')
                #     elif actor.type_id == 'traffic.stop':
                #         plt.plot(loc.x, loc.y, 'm*')
                #     elif 'speed_limit' in actor.type_id:
                #         plt.plot(loc.x, loc.y, 'md')
                #     elif actor.type_id == 'traffic.traffic_light':
                #         if actor.state == carla.TrafficLightState.Green:
                #             color = 'g'
                #         elif actor.state == carla.TrafficLightState.Red:
                #             color = 'r'
                #         elif actor.state == carla.TrafficLightState.Yellow:
                #             color = 'y'
                #         plt.plot(loc.x, loc.y, '8', color=color)
                # plt.show()

                snapshot_dict = process_world_snapshot(snapshot, world)

                if num_frames_saved == 1:
                    with open('snapshot_example.json', 'w') as f:
                        f.write(json.dumps(snapshot_dict, indent=4))

                # debug = world.debug
                

                # for actor_snapshot in snapshot:
                #     actual_actor = world.get_actor(actor_snapshot.id)
                #     if actual_actor.type_id == 'traffic.stop':
                #         debug.draw_box(carla.BoundingBox(actor_snapshot.get_transform().location,carla.Vector3D(0.5,0.5,2)),actor_snapshot.get_transform().rotation, 0.05, carla.Color(255,0,0,0),0)


                for col_idx, camera_id in  enumerate(['center', 'left', 'right', 'far']):
                    for row_idx, suffix in enumerate(['', 'seg', 'depth']):
                        img_name = 'image' + '_' + camera_id

                        if len(suffix) > 0:
                            img_name += '_' + suffix

                        img = locals()[img_name]

                        if suffix == 'seg':
                            img.convert(carla.ColorConverter.CityScapesPalette)
                        elif suffix == 'depth':
                            img.convert(carla.ColorConverter.LogarithmicDepth)

                        img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
                        img_array = np.reshape(img_array, (img.height, img.width, 4))
                        img_array = img_array[:, :, :3]

                        if suffix == '':
                            img_array = cv2.resize(img_array, (480, 360), cv2.INTER_CUBIC)
                        elif suffix == 'seg':
                            img_array = cv2.resize(img_array, (480, 360), cv2.INTER_NEAREST)
                        elif suffix == 'depth':
                            img_array = cv2.resize(img_array, (480, 360), cv2.INTER_NEAREST)
                        else:
                            raise ValueError("Unhandled image type: %s" % suffix)

                        xmin = col_idx * 480
                        xmax = (col_idx + 1) * 480
                        ymin = row_idx * 360
                        ymax = (row_idx + 1) * 360

                        mosaic_array[ymin:ymax, xmin:xmax, :] = img_array

                cv2.imshow('mosaic', mosaic_array); cv2.waitKey(1)


                # Draw the display.
                # img_array = np.zeros((360, 1440, 3), dtype=np.uint8)

                # center_array = np.frombuffer(image_center.raw_data, dtype=np.uint8)
                # center_array = np.reshape(center_array, (image_center.height, image_center.width, 4))
                # center_array = center_array[:, :, :3]
                # center_array = cv2.resize(center_array, (480, 360))

                # left_array = np.frombuffer(image_left.raw_data, dtype=np.uint8)
                # left_array = np.reshape(left_array, (image_left.height, image_left.width, 4))
                # left_array = left_array[:, :, :3]
                # left_array = cv2.resize(left_array, (480, 360))

                # right_array = np.frombuffer(image_right.raw_data, dtype=np.uint8)
                # right_array = np.reshape(right_array, (image_right.height, image_right.width, 4))
                # right_array = right_array[:, :, :3]
                # right_array = cv2.resize(right_array, (480, 360))

                # far_array = np.frombuffer(image_far.raw_data, dtype=np.uint8)
                # far_array = np.reshape(far_array, (image_far.height, image_far.width, 4))
                # far_array = far_array[:, :, :3]
                # far_array = cv2.resize(far_array, (480, 360))

                # img_array[:, :480, :]    = left_array
                # img_array[:, 480:960, :] = center_array
                # img_array[:, 960:, :]    = right_array 
                # cv2.imshow('fused', img_array);
                # cv2.imshow('far', far_array)
                # cv2.waitKey(1)

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