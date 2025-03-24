import time
import jsbsim
import matplotlib.pyplot as plt
import numpy as np

DT = 0.0083333
altitudes = []
airspeeds = []

def main():
    # Initialize JSBSim
    sim = jsbsim.FGFDMExec('../..')
    sim.load_model('737')  # Load the B747 model
    # sim.load_model('A320')  # Load the A320 model
    sim.set_dt(DT)  # Set the simulation time step to DT seconds

    # Set initial conditions
    sim['ic/h-sl-ft'] = 0  # Initial altitude in feet
    sim['ic/terrain-elevation-ft'] = 0  # Terrain elevation in feet
    sim['ic/vc-kts'] = 0  # Initial airspeed in knots
    sim['ic/psi-true-deg'] = 0  # Initial heading in degrees
    sim['ic/gamma-deg'] = 0  # Initial flight path angle in degrees
    sim['ic/alpha-deg'] = 0  # Initial angle of attack in degrees

    # Initialize the simulation
    sim.run_ic()

    # Example commands to control the aircraft
    commands = [
        {'time': 10.0, 'fcs/steer-cmd-norm': 0.5},
        {'time': 10.0, 'fcs/elevator-cmd-norm': 0.5},
        {'time': 12.0, 'fcs/steer-cmd-norm': 0},
        {'time': 12.0, 'fcs/elevator-cmd-norm': 0},
        {'time': 20.0, 'propulsion/starter_cmd': 1},
        {'time': 25.0, 'propulsion/cutoff_cmd': 0},
        {'time': 25.0, 'fcs/throttle-cmd-norm[0]': 1.0},
        {'time': 25.0, 'fcs/throttle-cmd-norm[1]': 1.0},
        {'time': 55.0, 'fcs/steer-cmd-norm': 1},
        {'time': 60.0, 'fcs/steer-cmd-norm': 0},
        {'time': 200.0, 'fcs/throttle-cmd-norm[0]': 0.9},
        {'time': 200.0, 'fcs/throttle-cmd-norm[1]': 0.9}
    ]


    command_index = 0
    loop_counter = 0
    sim_time = 0.0
    total_time = 400.0
    command = ""
    while sim_time < total_time:
        if command_index < len(commands) and command == "":
            command = commands[command_index]
            command_index += 1

        if type(command) is dict and sim.get_sim_time() >= command['time']:
            for key, value in command.items():
                if key != 'time':
                    sim[key] = value
                    print(f"Setting {key} to {value}")
            command = ""

        # Run the simulation for one time step
        sim.run()
        sim_time = sim.get_sim_time()

        # Print the current state of the aircraft
        if loop_counter % 100 == 0:
            print(f"Time: {sim.get_sim_time():.2f}, Altitude: {sim['position/h-sl-ft']:.2f}, Airspeed: {sim['velocities/vc-kts']:.2f}") 

        loop_counter += 1

        altitudes.append(sim['position/h-sl-ft'])
        airspeeds.append(sim['velocities/vc-kts'])

        # Wait for the next time step
        # time.sleep(DT)
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

if __name__ == "__main__":
    main()