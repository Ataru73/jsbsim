import time
import jsbsim
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from trajectory_controller import TrajectoryController, go_to_position

"""
JSBSim Pilot Script
==================

This script simulates piloting either an F15 fighter jet or an F450 quadrotor drone.

Usage:
    python Pilot.py [options]

Options:
    drone    - Use F450 quadrotor drone (default: F15)
    f15      - Use F15 fighter jet (default)
    gui      - Enable FlightGear GUI visualization

Examples:
    python Pilot.py                    # Run F15 simulation
    python Pilot.py drone              # Run drone simulation
    python Pilot.py drone gui          # Run drone simulation with GUI
    python Pilot.py f15 gui            # Run F15 simulation with GUI
"""

DT = 0.0083333
altitudes = []
airspeeds = []
positions_x = []  # North-South position in feet
positions_y = []  # East-West position in feet

class TrajectoryController:
    """Controller for planning and executing trajectories to target positions.
    
    Supports both single waypoint and multi-waypoint trajectories:
    - Single point: set_target_position(x, y, z, target_time)
    - Multiple waypoints: set_waypoints([(x1, y1, z1, t1), (x2, y2, z2, t2), ...])
    """
    
    def __init__(self, sim, dt=0.0083333):
        self.sim = sim
        self.dt = dt
        
        # Single waypoint mode
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 320.0
        self.target_time = 0.0
        self.start_time = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_z = 0.0
        self.active = False
        
        # Multi-waypoint mode
        self.waypoints = []  # List of (x, y, z, time) tuples
        self.current_waypoint_index = 0
        self.waypoint_mode = False
        
        # PID controller gains for position control (tuned for F450)
        # Extremely conservative gains to prevent overshoot
        
        # X-axis (Roll/East-West) - extremely conservative for stability
        self.kp_x = 0.002   # Proportional gain for x-direction - very small
        self.ki_x = 0.00005  # Integral gain for x-direction - very small
        self.kd_x = 0.001   # Derivative gain for x-direction - very small
        
        # Y-axis (Pitch/North-South) - symmetric gains with X-axis
        self.kp_y = 0.002   # Proportional gain for y-direction - same as X for symmetry
        self.ki_y = 0.00005  # Integral gain for y-direction - same as X for symmetry
        self.kd_y = 0.001   # Derivative gain for y-direction - same as X for symmetry
        
        # Feedforward compensation factors - minimal for stability
        self.ff_gain_x = 0.0005  # Feedforward gain for X-axis - very small
        self.ff_gain_y = 0.0005  # Feedforward gain for Y-axis - same as X for symmetry
        
        # Velocity-based damping - light for both axes
        self.vel_damping_x = 0.001  # Velocity damping for X-axis - very small
        self.vel_damping_y = 0.001  # Velocity damping for Y-axis - same as X
        
        # Overshoot prevention parameters (relaxed for better tracking)
        self.y_overshoot_threshold = 50.0  # Distance threshold to activate overshoot prevention
        self.y_brake_gain = 0.3  # Gentle braking when approaching target in Y
        
        # Error accumulation for integral control
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
        # Control limits (for pilot inputs) - extremely conservative for stability
        self.max_roll_cmd = 0.02   # Maximum roll command (normalized) - extremely small
        self.max_pitch_cmd = 0.02  # Maximum pitch command (normalized) - extremely small
        
        # Target altitude control
        self.target_altitude = 320.0  # Target altitude in feet (higher for safety)
        self.base_heave_cmd = 0.2     # Base heave command for hover
        
    def set_target_position(self, x, y, target_time):
        """Set a target position (x, y) to reach at the specified time.
        
        Args:
            x (float): Target x-position in feet (East-West)
            y (float): Target y-position in feet (North-South)
            target_time (float): Time to reach the target position
        """
        self.target_x = x
        self.target_y = y
        self.target_time = target_time
        self.start_time = self.sim.get_sim_time()
        
        # Record current position as starting point
        self.start_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
        self.start_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
        
        # Reset PID controller state
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
        self.active = True
        
        print(f"Target set: ({x:.1f}, {y:.1f}) ft at time {target_time:.1f}s")
        print(f"Current position: ({self.start_x:.1f}, {self.start_y:.1f}) ft at time {self.start_time:.1f}s")
        print(f"Distance to target: {((x - self.start_x)**2 + (y - self.start_y)**2)**0.5:.1f} ft")
        print(f"Time available: {target_time - self.start_time:.1f}s")
        
    def calculate_desired_position(self, current_time):
        """Calculate the desired position at the current time based on linear interpolation."""
        if not self.active or current_time < self.start_time:
            return self.start_x, self.start_y
            
        # Calculate time progress (0 to 1)
        time_progress = (current_time - self.start_time) / (self.target_time - self.start_time)
        time_progress = max(0.0, min(1.0, time_progress))  # Clamp to [0, 1]
        
        # Linear interpolation between start and target positions
        desired_x = self.start_x + (self.target_x - self.start_x) * time_progress
        desired_y = self.start_y + (self.target_y - self.start_y) * time_progress
        
        return desired_x, desired_y
        
    def update_control_commands(self):
        """Update the control commands to track the desired trajectory."""
        if not self.active:
            return
            
        current_time = self.sim.get_sim_time()
        
        # Check if we've reached the target time
        if current_time >= self.target_time:
            self.active = False
            # Clear all control commands to stop the drone
            self.sim['fcs/aileron-cmd-norm'] = 0.0
            self.sim['fcs/elevator-cmd-norm'] = 0.0
            # Keep altitude hold active
            altitude_error = self.target_altitude - self.sim['position/h-sl-ft']
            heave_adjustment = altitude_error * 0.01
            heave_cmd = self.base_heave_cmd + heave_adjustment
            heave_cmd = max(-0.5, min(1.0, heave_cmd))
            self.sim['fcs/cmdHeave_nd'] = heave_cmd
            print(f"Target time reached at {current_time:.1f}s")
            return
            
        # Get current position
        current_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
        current_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
        current_alt = self.sim['position/h-sl-ft']
        
        # Calculate desired position based on trajectory
        desired_x, desired_y = self.calculate_desired_position(current_time)
        
        # Calculate position errors
        error_x = desired_x - current_x
        error_y = desired_y - current_y
        
        # Get current velocities for velocity damping
        vel_x = self.sim['velocities/v-fps']  # East-West velocity
        vel_y = self.sim['velocities/u-fps']  # North-South velocity
        
        # Calculate desired velocities for feedforward control
        time_progress = (current_time - self.start_time) / (self.target_time - self.start_time)
        time_progress = max(0.0, min(1.0, time_progress))
        
        # Calculate desired velocity based on trajectory slope
        if self.target_time > self.start_time:
            desired_vel_x = (self.target_x - self.start_x) / (self.target_time - self.start_time)
            desired_vel_y = (self.target_y - self.start_y) / (self.target_time - self.start_time)
        else:
            desired_vel_x = 0.0
            desired_vel_y = 0.0
        
        # PID control for x-direction (East-West, affects roll)
        self.error_x_integral += error_x * self.dt
        # Add integral windup protection
        self.error_x_integral = max(-50, min(50, self.error_x_integral))
        error_x_derivative = (error_x - self.prev_error_x) / self.dt
        
        # Enhanced roll command with feedforward and velocity damping
        roll_cmd = (self.kp_x * error_x + 
                   self.ki_x * self.error_x_integral + 
                   self.kd_x * error_x_derivative +
                   self.ff_gain_x * desired_vel_x -
                   self.vel_damping_x * vel_x)
        
        # PID control for y-direction (North-South, affects pitch)
        self.error_y_integral += error_y * self.dt
        # Add integral windup protection
        self.error_y_integral = max(-50, min(50, self.error_y_integral))
        error_y_derivative = (error_y - self.prev_error_y) / self.dt
        
        # Enhanced pitch command with feedforward and velocity damping
        pitch_cmd = (self.kp_y * error_y + 
                    self.ki_y * self.error_y_integral + 
                    self.kd_y * error_y_derivative +
                    self.ff_gain_y * desired_vel_y -
                    self.vel_damping_y * vel_y)
        
        # Advanced overshoot prevention for Y-axis
        distance_to_target_y = abs(error_y)
        if distance_to_target_y < self.y_overshoot_threshold:
            # Check if we're moving towards the target with significant velocity
            if (error_y > 0 and vel_y > 0.5) or (error_y < 0 and vel_y < -0.5):
                # Apply aggressive braking
                brake_force = -self.y_brake_gain * vel_y
                pitch_cmd += brake_force
                
        # Additional safety: if we're very close to target, limit command authority
        if distance_to_target_y < 5.0:  # Within 5 feet
            pitch_cmd *= 0.3  # Reduce command authority to 30%
        
        # Limit control commands
        roll_cmd = max(-self.max_roll_cmd, min(self.max_roll_cmd, roll_cmd))
        pitch_cmd = max(-self.max_pitch_cmd, min(self.max_pitch_cmd, pitch_cmd))
        
        # Apply control commands to the drone using pilot inputs
        # For F450 drone, use aileron and elevator commands (normalized)
        self.sim['fcs/aileron-cmd-norm'] = roll_cmd    # Roll command
        self.sim['fcs/elevator-cmd-norm'] = pitch_cmd  # Pitch command
        
        # Maintain altitude with heave control
        altitude_error = self.target_altitude - current_alt
        heave_adjustment = altitude_error * 0.01  # Small gain for altitude control
        heave_cmd = self.base_heave_cmd + heave_adjustment
        heave_cmd = max(-0.5, min(1.0, heave_cmd))  # Limit heave command
        self.sim['fcs/cmdHeave_nd'] = heave_cmd
        
        # Update previous errors for next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # Debug output every 2 seconds
        if int(current_time * 10) % 20 == 0:  # Every 2 seconds
            print(f"T={current_time:.1f}s: Pos=({current_x:.1f},{current_y:.1f}), "
                  f"Desired=({desired_x:.1f},{desired_y:.1f}), "
                  f"Error=({error_x:.1f},{error_y:.1f}), "
                  f"Cmd=(R:{roll_cmd:.3f},P:{pitch_cmd:.3f})")

def go_to_position(x, y, target_time):
    """Function to generate control commands to pilot the drone to position (x,y) at time t.
    
    Args:
        x (float): Target x-position in feet (East-West direction)
        y (float): Target y-position in feet (North-South direction)
        target_time (float): Time in seconds to reach the target position
        
    Returns:
        TrajectoryController: Controller object that can be used to execute the trajectory
        
    Example:
        # Create a controller to go to position (100, 50) at 15 seconds
        controller = go_to_position(100, 50, 15.0)
        
        # In the main simulation loop, call:
        # controller.update_control_commands()
    """
    # This function would typically be called with an existing sim object
    # For now, we'll return a factory function that creates the controller
    def create_controller(sim):
        controller = TrajectoryController(sim)
        controller.set_target_position(x, y, target_time)
        return controller
    
    return create_controller

def main():
    # Parse command line arguments
    aircraft_type = 'f15'  # Default to F15
    gui = False
    
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        return
    
    # Parse arguments
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == 'gui':
                gui = True
            elif arg == 'drone':
                aircraft_type = 'F450'  # Quadrotor drone
            elif arg == 'f15':
                aircraft_type = 'f15'
            else:
                print(f"Unknown argument: {arg}")
                print("Use 'python Pilot.py help' for usage information")
                return
    
    # Initialize JSBSim
    sim = jsbsim.FGFDMExec('../..')
    sim.load_model(aircraft_type)  # Load the selected aircraft model
    sim.set_dt(DT)  # Set the simulation time step to DT seconds
    
    print(f"Loading aircraft: {aircraft_type}")
    is_drone = aircraft_type == 'F450'

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
    
    if is_drone:
        # Drone-specific initial conditions
        sim['ic/h-agl-ft'] = 10           # Start 10 ft above ground
        sim['ic/h-sl-ft'] = 170           # Start 10 ft above the 160 ft terrain = 170 ft sea level
        sim['ic/psi-true-deg'] = 0.0      # Heading in degrees
        sim['ic/u-fps'] = 0.0             # Initial forward velocity
        sim['ic/v-fps'] = 0.0             # Initial side velocity
        sim['ic/w-fps'] = 0.0             # Initial vertical velocity
    else:
        # F15 initial conditions
        sim['ic/h-agl-ft'] = 5            # Match altitude above ground level
        sim['ic/altitude-gnd-ft'] = 10     # Ground altitude in feet
        sim['ic/psi-true-deg'] = 178.0    # Heading in degrees

    # Initialize the simulation
    sim.run_ic()
    
    # Trim the drone for stable hover if it's a drone
    if is_drone:
        sim.do_trim(2)  # Trim for steady state flight
    
    # Define commands based on aircraft type
    if is_drone:
        # Drone-specific commands for quadrotor control
        # Use stable hover configuration
        commands = [
            {'time': 0.1, 'fcs/ScasEngage': 1},             # Enable stability control system
            {'time': 0.1, 'fcs/cmdHeave_nd': 0.3},          # Stronger heave command for stable hover
            {'time': 0.1, 'fcs/throttle-cmd-norm[0]': 0.6}, # More conservative throttle for stability
            {'time': 0.1, 'fcs/throttle-cmd-norm[1]': 0.6},
            {'time': 0.1, 'fcs/throttle-cmd-norm[2]': 0.6},
            {'time': 0.1, 'fcs/throttle-cmd-norm[3]': 0.6},
            {'time': 2.0, 'fcs/cmdHeave_nd': 0.2},          # Maintain hover
            {'time': 5.0, 'fcs/cmdHeave_nd': 0.15},         # Stable hover
            {'time': 10.0, 'fcs/aileron-cmd-norm': 0.02},   # Very gentle roll
            {'time': 13.0, 'fcs/aileron-cmd-norm': 0.0},    # Level out
            {'time': 15.0, 'fcs/elevator-cmd-norm': 0.02},  # Very gentle pitch
            {'time': 18.0, 'fcs/elevator-cmd-norm': 0.0},   # Level out
            {'time': 20.0, 'fcs/rudder-cmd-norm': 0.02},    # Very gentle yaw
            {'time': 23.0, 'fcs/rudder-cmd-norm': 0.0},     # Stop yaw
            {'time': 25.0, 'fcs/cmdHeave_nd': 0.0},         # Gentle descent
            {'time': 28.0, 'fcs/cmdHeave_nd': -0.05},       # Very gentle descent
        ]
        total_time = 30.0  # Shorter simulation time for drone
    else:
        # F15 commands for conventional aircraft
        commands = [
            {'time': 0.1, 'fcs/flap-cmd-norm': 0.3},  # Deploy flaps for takeoff
            {'time': 0.2, 'fcs/break-cmd-norm': 0.0},  # Release brakes
            {'time': 0.3, 'propulsion/starter_cmd': 1},  # Start engines
            {'time': 0.4, 'propulsion/cutoff_cmd': 0},  # Ensure engines are running
            {'time': 1.0, 'fcs/throttle-cmd-norm[0]': 1.0},  # Full throttle for takeoff
            {'time': 1.0, 'fcs/throttle-cmd-norm[1]': 1.0},
            {'time': 10.0, 'fcs/elevator-cmd-norm': -0.5},  # Pitch up for takeoff
            {'time': 45.0, 'fcs/flap-cmd-norm': 0.0},  # Retract flaps
            {'time': 46.0, 'gear/gear-cmd-norm': 0},  # Retract landing gear
            {'time': 47.0, 'fcs/throttle-cmd-norm[0]': 0.5, 'fcs/throttle-cmd-norm[1]': 0.5},  # Half throttle for climb
            {'time': 50.0, 'ap/heading_hold': 1, 'ap/heading_setpoint': 180, 'ap/altitude_hold': 1, 'ap/altitude_setpoint': 2500},  
            {'time': 60.0, 'fcs/elevator-cmd-norm': -0.05}
        ]
        total_time = 100.0  # Standard simulation time for F15

    # sort commands by time
    commands = sorted(commands, key=lambda x: x['time'])

    command_index = 0
    loop_counter = 0
    sim_time = 0.0
    command = ""
    while sim_time < total_time:
        if command_index < len(commands) and command == "":
            command = commands[command_index]
            command_index += 1

        if type(command) is dict and sim.get_sim_time() >= command['time']:
            for key, value in command.items():
                if key != 'time' and key != 'trim':
                    sim[key] = value
                    print(f"Setting {key} to {value}")
                elif key == 'trim':
                    sim.do_trim(int(value))  # Convert to int for trim mode
            command = ""

        if is_drone:
            # Drone-specific altitude control - very gentle
            # Just maintain basic stability without aggressive control
            pass
        else:
            # F15 altitude control
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
        positions_x.append(sim['position/distance-from-start-lon-mt'] * 3.28084)  # Convert meters to feet
        positions_y.append(sim['position/distance-from-start-lat-mt'] * 3.28084)  # Convert meters to feet

        if gui:
            # Wait for the next time step
            time.sleep(DT)

    knots_per_sec = np.array(airspeeds) * 3600 / 1852  # Convert knots to meters per second
    accelerations = np.gradient(np.array(knots_per_sec), np.arange(len(knots_per_sec))*DT) * (9.81 / 1800)

    # Determine aircraft name for plots
    aircraft_name = "F450 Drone" if is_drone else "F15"

    # Plot the altitude and airspeed
    plt.plot([i*DT for i in range(len(altitudes))], altitudes, label='Altitude')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Altitude (ft)')
    plt.title(f'{aircraft_name} Altitude Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('altitude_plot.png')  # Save the altitude plot to a file
    plt.close()  # Close the plot to free up memory

    plt.plot([i*DT for i in range(len(altitudes))], airspeeds, label='Airspeed')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Airspeed (kts)')
    plt.title(f'{aircraft_name} Airspeed Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('airspeed_plot.png')  # Save the airspeed plot to a file
    plt.close()  # Close the plot to free up memory

    # Plot acceleration
    plt.plot([i*DT for i in range(len(accelerations))], accelerations, label='Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (ft/s^2)')
    plt.title(f'{aircraft_name} Acceleration Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('acceleration_plot.png')  # Save the acceleration plot to a file
    plt.close()  # Close the plot to free up memory
    
    # Plot ground trajectory (top-down view)
    plt.figure(figsize=(10, 8))
    plt.plot(positions_x, positions_y, 'b-', linewidth=2, label='Flight Path')
    plt.plot(positions_x[0], positions_y[0], 'go', markersize=10, label='Start')
    plt.plot(positions_x[-1], positions_y[-1], 'ro', markersize=10, label='End')
    
    # Add time markers every 5 seconds
    time_markers = np.arange(0, len(positions_x), int(5.0 / DT))  # Every 5 seconds
    for i, marker_idx in enumerate(time_markers):
        if marker_idx < len(positions_x):
            plt.plot(positions_x[marker_idx], positions_y[marker_idx], 'ko', markersize=4)
            plt.annotate(f'{i*5}s', (positions_x[marker_idx], positions_y[marker_idx]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('East-West Position (ft)')
    plt.ylabel('North-South Position (ft)')
    plt.title(f'{aircraft_name} Ground Trajectory (Top-Down View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio equal for accurate representation
    plt.savefig('ground_trajectory_plot.png')  # Save the ground trajectory plot to a file
    plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    main()