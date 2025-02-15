"""Example script to generate traffic in the simulation"""
import os
import sys
import time
from numpy import random

from ..utils import color_error_string


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def generate_traffic(carla_ip, rpc_port, tm_port, number_of_vehicles, number_of_walkers, traffic_manager_is_up, logs_path,
                     tm_ready_to_warm_up, tm_ready_to_take_data, dt_ready_to_warm_up, dt_ready_to_take_data,
                     dt_want_to_stop_taking_data, wait_a_little_bit_before_starting, warm_up_frames,
                     hero=True):
    try:
        import carla
    except:
        raise Exception(color_error_string(f"Not able to import Carla!"))
    
    if not os.path.isdir(os.path.dirname(logs_path)):
        try:
            os.mkdir(os.path.dirname(logs_path))
        except:
            Exception(color_error_string(f"Unable to find out log dir! [{os.path.dirname(logs_path)}]"))
    if os.path.exists(logs_path):
        os.remove(logs_path)
    with open(logs_path, 'w') as _:
        pass
    
    out_err_logs_file = open(logs_path, "w")
    sys.stdout = out_err_logs_file
    sys.stderr = out_err_logs_file

    vehicles_list = []
    walkers_list = []
    all_id = []

    client = carla.Client(carla_ip, rpc_port)
    client.set_timeout(1000.0)
    synchronous_master = False
    random.seed(int(time.time()))

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(tm_port)

        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)


        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False

        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = f"requested {number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points"
            print(msg)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        if hero:
            blueprint = world.get_blueprint_library().find('vehicle.ford.mustang')
            blueprint.set_attribute('color', blueprint.get_attribute('color').recommended_values[0])
            blueprint.set_attribute('role_name', 'hero')
            batch.append(SpawnActor(blueprint, spawn_points[random.randint(0, len(spawn_points) - 1)])
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        for n, transform in enumerate(spawn_points[:]):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                print(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        traffic_manager_is_up.set()
        sys.stdout.flush()
        sys.stderr.flush()

        # Let's gt the hero actor
        possible_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in possible_vehicles:
            if vehicle.attributes['role_name'] == 'hero':
                print("Ego vehicle found")
                hero_actor = vehicle
                break

        # Pre-Warm UP
        for i in range(100):
            world.tick()

        tm_ready_to_warm_up.set()
        while True:
            if dt_ready_to_warm_up.is_set():
                break
        time.sleep(wait_a_little_bit_before_starting)
        for i in range(warm_up_frames):
            world.tick()
            hero_transform = hero_actor.get_transform()
            hero_transform.location.z += 30
            hero_transform.rotation.pitch = -90.
            world.get_spectator().set_transform(hero_transform)
        tm_ready_to_take_data.set()
        while True:
            if dt_ready_to_take_data.is_set():
                break
        time.sleep(wait_a_little_bit_before_starting)
        while True:
            world.tick()
            hero_transform = hero_actor.get_transform()
            hero_transform.location.z += 30
            hero_transform.rotation.pitch = -90.
            world.get_spectator().set_transform(hero_transform)
            if dt_want_to_stop_taking_data.is_set():
                break
        while True:
            # We need to tick sometimes otherwise the process handler thinks that carla is died!
            time.sleep(5)
            world.tick()
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)
