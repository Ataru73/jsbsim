import time
import jsbsim
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
import pygame  # Add this import for joystick support

DT = 0.0083333
altitudes = []
airspeeds = []

def main():
    # Initialize pygame for joystick input
    pygame.init()
    pygame.joystick.init()

    # Check if a joystick is connected
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Please connect a joystick.")
        return

    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")

    # Initialize JSBSim
    sim = jsbsim.FGFDMExec('../..')
    sim.load_model('f15')  # Load the B747 model
    # sim.load_model('A320')  # Load the A320 model
    sim.set_dt(DT)  # Set the simulation time step to DT seconds

    gui = False
    # If the 1st argument is 'gui', enable the GUI
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        gui = True

    if gui:
        # Enable FlightGear output
        sim.set_output_directive('data_output/flightgear.xml')
        

    # Set latitude and longitude to the runway of Pratica di Mare
    lat = 41.658
    lon = 12.446

    # Set initial conditions for JSBSim
    sim['ic/lat-gc-deg'] = lat      # Latitude of Pratica di Mare
    sim['ic/long-gc-deg'] = lon      # Longitude of Pratica di Mare
    sim['ic/terrain-elevation-ft'] = 160  # Flat terrain
    # sim['ic/h-sl-ft'] = 50             # Start at 50 ft above sea level
    sim['ic/h-agl-ft'] = 5            # Match altitude above ground level    sim['ic/psi-true-deg'] = 310.5  # Heading in degrees
    # sim['ic/altitude-ft'] = 39      # Altitude in feet
    sim['ic/altitude-gnd-ft'] = 10     # Ground altitude in feet
    sim['ic/psi-true-deg'] = 178.0  # Heading in degrees

    # Initialize the simulation
    sim.run_ic()

    # Example commands to control the aircraft
    # commands = [
    #     {'time': 10.0, 'fcs/steer-cmd-norm': 0.5},
    #     {'time': 10.0, 'fcs/elevator-cmd-norm': 0.5},
    #     {'time': 12.0, 'fcs/steer-cmd-norm': 0},
    #     {'time': 15.0, 'fcs/flap-cmd-norm': 0.3},
    #     {'time': 16.0, 'fcs/break-cmd-norm': 0.0},
    #     {'time': 20.0, 'propulsion/starter_cmd': 1},
    #     {'time': 25.0, 'propulsion/cutoff_cmd': 0},
    #     {'time': 25.0, 'fcs/throttle-cmd-norm[0]': 1.0},
    #     {'time': 25.0, 'fcs/throttle-cmd-norm[1]': 1.0},
    #     {'time': 30.0, 'fcs/elevator-cmd-norm': 0.5},
    #     {'time': 150.0, 'fcs/elevator-cmd-norm': -0.1},
    #     {'time': 130.0, 'fcs/flap-cmd-norm': 0.1},
    #     {'time': 180.0, 'gear/gear-cmd-norm': 0},
    #     {'time': 200.0, 'pressurize-cmd': 1},
    #     {'time': 200.0, 'fcs/throttle-cmd-norm[0]': 0.9},
    #     {'time': 200.0, 'fcs/throttle-cmd-norm[1]': 0.9},
    #     {'time': 200.0, 'fcs/flap-cmd-norm': 0},
    #     {'time': 200.0, 'fcs/ele-cmd-norm': 0.2},
    #     {'time': 210.0, 'fcs/elevator-cmd-norm': 1.0},
    #     {'time': 275.0, 'fcs/leveling-cmd-norm': 0}
    # ]

    auto_commands = [
        {'time': 0.1, 'fcs/flap-cmd-norm': 0.3},  # Deploy flaps for takeoff
        {'time': 0.2, 'fcs/break-cmd-norm': 0.0},  # Release brakes
        {'time': 0.3, 'propulsion/starter_cmd': 1},  # Start engines
        {'time': 0.4, 'propulsion/cutoff_cmd': 0},  # Ensure engines are running
        # {'time': 1.0, 'fcs/throttle-cmd-norm[0]': 1.0},  # Full throttle for takeoff
        # {'time': 1.0, 'fcs/throttle-cmd-norm[1]': 1.0},
        # {'time': 10.0, 'fcs/elevator-cmd-norm': -0.2},  # Pitch up for takeoff
        # # {'time': 40.0, 'fcs/elevator-cmd-norm': 0.0},  # Reduce pitch for climb
        # {'time': 45.0, 'fcs/flap-cmd-norm': 0.0},  # Retract flaps
        # {'time': 46.0, 'gear/gear-cmd-norm': 0},  # Retract landing gear
        # {'time': 47.0, 'fcs/throttle-cmd-norm[0]': 0.8, 'fcs/throttle-cmd-norm[1]': 0.8}, # Lower throttle for climb  
        # {'time': 50.0, 'ap/heading_hold': 1, 'ap/heading_setpoint': 180, 'ap/altitude_hold': 1, 'ap/altitude_setpoint': 2500},  
        # {'time': 51.0, 'fcs/elevator-cmd-norm': 0.0},  # Level off
        # {'time': 60.0, 'fcs/elevator-cmd-norm': -0.05}
    ]

    # sort commands by time
    auto_commands = sorted(auto_commands, key=lambda x: x['time'])

    command_index = 0
    loop_counter = 0
    sim_time = 0.0
    total_time = 600.0
    command = ""
    while sim_time < total_time:
        # Process joystick events
        pygame.event.pump()

        # Read joystick axes (example: throttle and elevator)
        throttle = joystick.get_axis(2)  # Axis 1 for throttle (adjust based on your joystick)
        elevator = joystick.get_axis(1)  # Axis 3 for elevator (adjust based on your joystick)
        roll = joystick.get_axis(0)  # Axis 0 for roll (adjust based on your joystick)
        yaw = joystick.get_axis(3)  # Axis 2 for yaw (adjust based on your joystick)

        # Map joystick values (-1 to 1) to JSBSim command ranges
        sim['fcs/throttle-cmd-norm[0]'] = (1 - throttle) / 2  # Map -1 to 1 -> 0 to 1
        sim['fcs/throttle-cmd-norm[1]'] = (1 - throttle) / 2
        sim['fcs/elevator-cmd-norm'] = -elevator  # Invert elevator axis if needed
        sim['fcs/aileron-cmd-norm'] = roll  # Roll control
        sim['fcs/rudder-cmd-norm'] = -yaw  # Yaw control

        # Print joystick values for debugging
        if loop_counter % 100 == 0:
            print(f"Joystick Throttle: {throttle:.2f}, Elevator: {elevator:.2f}, Roll: {roll:.2f}, Yaw: {yaw:.2f}")

        if command_index < len(auto_commands) and command == "":
            command = auto_commands[command_index]
            command_index += 1

        if type(command) is dict and sim.get_sim_time() >= command['time']:
            for key, value in command.items():
                if key != 'time' and key != 'trim':
                    sim[key] = value
                    print(f"Setting {key} to {value}")
                elif key == 'trim':
                    sim.do_trim(value)
            command = ""

        if sim['position/h-sl-ft'] > 10000:
            if sim['attitude/pitch-rad'] > 0.0:
                pass 
                # sim['fcs/elevator-cmd-norm'] = -0.05
            elif sim['attitude/pitch-rad'] < 0.0:
                sim['fcs/elevator-cmd-norm'] = 0.05

        # Run the simulation for one time step
        sim.run()
        sim_time = sim.get_sim_time()

        # Print the current state of the aircraft
        if loop_counter % 100 == 0:
            print(f"Time: {sim.get_sim_time():.2f}, Altitude: {sim['position/h-sl-ft']:.2f}, Airspeed: {sim['velocities/vc-kts']:.2f}")
            print(f"Pitch: {sim['attitude/pitch-rad']:.2f}, Roll: {sim['attitude/roll-rad']:.2f}, Yaw: {sim['attitude/heading-true-rad']:.2f}")
            print()

        loop_counter += 1

        altitudes.append(sim['position/h-sl-ft'])
        airspeeds.append(sim['velocities/vc-kts'])

        if gui:
            # Wait for the next time step
            time.sleep(DT)

    knots_per_sec = np.array(airspeeds) * 3600 / 1852  # Convert knots to meters per second
    accelerations = np.gradient(np.array(knots_per_sec), np.arange(len(knots_per_sec))*DT) * (9.81 / 1800)

    # Plot the altitude and airspeed
    plt.plot([i*DT for i in range(len(altitudes))], altitudes, label='Altitude')
    plt.xlabel('Time')
    plt.ylabel('Altitude (ft)')
    # plt.ylim(0, 5000)
    plt.title('B737 Altitude Over Time')
    plt.legend()
    plt.savefig('altitude_plot.png')  # Save the altitude plot to a file
    plt.close()  # Close the plot to free up memory

    plt.plot([i*DT for i in range(len(altitudes))], airspeeds, label='Airspeed')
    plt.xlabel('Time')
    plt.ylabel('Airspeed (kts)')
    # plt.ylim(0, 300)
    plt.title('B737 Airspeed Over Time')
    plt.legend()
    plt.savefig('airspeed_plot.png')  # Save the airspeed plot to a file
    plt.close()  # Close the plot to free up memory

    # Plot acceleration
    plt.plot([i*DT for i in range(len(accelerations))], accelerations, label='Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (ft/s^2)')
    # plt.ylim(-10, 10)
    plt.title('B737 Acceleration Over Time')
    plt.legend()
    plt.savefig('acceleration_plot.png')  # Save the acceleration plot to a file
    plt.close()  # Close the plot to free up memory

    # Quit pygame when the simulation ends
    pygame.quit()

if __name__ == "__main__":
    main()