"""
Trajectory Controller Module
============================

This module provides trajectory control functionality for JSBSim aircraft,
specifically designed for quadrotor drone control but adaptable to other aircraft types.

The TrajectoryController class supports:
- Single waypoint navigation: set_target_position(x, y, z, target_time)
- Multi-waypoint trajectories: set_waypoints([(x1, y1, z1, t1), (x2, y2, z2, t2), ...])
- PID-based position control with feedforward compensation
- Velocity damping and overshoot prevention
- Altitude hold capability

Usage:
    from trajectory_controller import TrajectoryController, go_to_position
    
    # Single waypoint example
    controller = TrajectoryController(sim)
    controller.set_target_position(100, 50, 320, 25.0)
    
    # Multi-waypoint example
    waypoints = [(50, 25, 320, 10), (100, 50, 300, 20), (0, 0, 320, 30)]
    controller.set_waypoints(waypoints)
    
    # In simulation loop
    controller.update_control_commands()
"""

import math


class TrajectoryController:
    """Controller for planning and executing trajectories to target positions.
    
    Supports both single waypoint and multi-waypoint trajectories:
    - Single point: set_target_position(x, y, z, target_time)
    - Multiple waypoints: set_waypoints([(x1, y1, z1, t1), (x2, y2, z2, t2), ...])
    """
    
    def __init__(self, sim, dt=0.0083333):
        """Initialize the trajectory controller.
        
        Args:
            sim: JSBSim simulation object
            dt: Simulation time step in seconds (default: 0.0083333 = 120 Hz)
        """
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
        
        # Z-axis (Altitude) - separate gains for vertical control
        self.kp_z = 0.01    # Proportional gain for altitude
        self.ki_z = 0.0001  # Integral gain for altitude
        self.kd_z = 0.005   # Derivative gain for altitude
        
        # Feedforward compensation factors - minimal for stability
        self.ff_gain_x = 0.0005  # Feedforward gain for X-axis - very small
        self.ff_gain_y = 0.0005  # Feedforward gain for Y-axis - same as X for symmetry
        self.ff_gain_z = 0.001   # Feedforward gain for Z-axis
        
        # Velocity-based damping - light for all axes
        self.vel_damping_x = 0.001  # Velocity damping for X-axis - very small
        self.vel_damping_y = 0.001  # Velocity damping for Y-axis - same as X
        self.vel_damping_z = 0.002  # Velocity damping for Z-axis
        
        # Overshoot prevention parameters (relaxed for better tracking)
        self.y_overshoot_threshold = 50.0  # Distance threshold to activate overshoot prevention
        self.y_brake_gain = 0.3  # Gentle braking when approaching target in Y
        
        # Error accumulation for integral control
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.error_z_integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        # Control limits (for pilot inputs) - extremely conservative for stability
        self.max_roll_cmd = 0.02   # Maximum roll command (normalized) - extremely small
        self.max_pitch_cmd = 0.02  # Maximum pitch command (normalized) - extremely small
        
        # Default target altitude control
        self.target_altitude = 320.0  # Default target altitude in feet (higher for safety)
        self.base_heave_cmd = 0.2     # Base heave command for hover
        
    def set_target_position(self, x, y, z, target_time):
        """Set a target position (x, y, z) to reach at the specified time.
        
        Args:
            x (float): Target x-position in feet (East-West)
            y (float): Target y-position in feet (North-South)
            z (float): Target z-position in feet (altitude)
            target_time (float): Time to reach the target position
        """
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_altitude = z  # Update altitude target
        self.target_time = target_time
        self.start_time = self.sim.get_sim_time()
        
        # Record current position as starting point
        self.start_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
        self.start_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
        self.start_z = self.sim['position/h-sl-ft']
        
        # Reset PID controller state
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.error_z_integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        # Disable waypoint mode
        self.waypoint_mode = False
        self.active = True
        
        print(f"Target set: ({x:.1f}, {y:.1f}, {z:.1f}) ft at time {target_time:.1f}s")
        print(f"Current position: ({self.start_x:.1f}, {self.start_y:.1f}, {self.start_z:.1f}) ft at time {self.start_time:.1f}s")
        print(f"Distance to target: {((x - self.start_x)**2 + (y - self.start_y)**2 + (z - self.start_z)**2)**0.5:.1f} ft")
        print(f"Time available: {target_time - self.start_time:.1f}s")
        
    def set_waypoints(self, waypoints):
        """Set multiple waypoints to follow in sequence.
        
        Args:
            waypoints: List of (x, y, z, time) tuples representing waypoints
        """
        if not waypoints:
            print("Warning: Empty waypoints list provided")
            return
            
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.waypoint_mode = True
        
        # Reset PID controller state
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.error_z_integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        # Set the first waypoint as the current target
        if waypoints:
            first_waypoint = waypoints[0]
            self.target_x, self.target_y, self.target_z, self.target_time = first_waypoint
            self.target_altitude = self.target_z
            self.start_time = self.sim.get_sim_time()
            
            # Record current position as starting point
            self.start_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
            self.start_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
            self.start_z = self.sim['position/h-sl-ft']
            
            self.active = True
            
            print(f"Waypoint mission started with {len(waypoints)} waypoints")
            print(f"First target: ({self.target_x:.1f}, {self.target_y:.1f}, {self.target_z:.1f}) ft at time {self.target_time:.1f}s")
        
    def _advance_to_next_waypoint(self):
        """Advance to the next waypoint in the sequence."""
        if not self.waypoint_mode or self.current_waypoint_index >= len(self.waypoints) - 1:
            # No more waypoints, stop
            self.active = False
            print("Waypoint mission completed")
            return
            
        # Move to next waypoint
        self.current_waypoint_index += 1
        next_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Update target and start positions
        self.start_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
        self.start_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
        self.start_z = self.sim['position/h-sl-ft']
        self.start_time = self.sim.get_sim_time()
        
        self.target_x, self.target_y, self.target_z, self.target_time = next_waypoint
        self.target_altitude = self.target_z
        
        # Reset integral errors for clean transition
        self.error_x_integral = 0.0
        self.error_y_integral = 0.0
        self.error_z_integral = 0.0
        
        print(f"Advancing to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}")
        print(f"Next target: ({self.target_x:.1f}, {self.target_y:.1f}, {self.target_z:.1f}) ft at time {self.target_time:.1f}s")
        
    def calculate_desired_position(self, current_time):
        """Calculate the desired position at the current time based on linear interpolation."""
        if not self.active or current_time < self.start_time:
            return self.start_x, self.start_y, self.start_z
            
        # Calculate time progress (0 to 1)
        time_progress = (current_time - self.start_time) / (self.target_time - self.start_time)
        time_progress = max(0.0, min(1.0, time_progress))  # Clamp to [0, 1]
        
        # Linear interpolation between start and target positions
        desired_x = self.start_x + (self.target_x - self.start_x) * time_progress
        desired_y = self.start_y + (self.target_y - self.start_y) * time_progress
        desired_z = self.start_z + (self.target_z - self.start_z) * time_progress
        
        return desired_x, desired_y, desired_z
        
    def update_control_commands(self):
        """Update the control commands to track the desired trajectory."""
        if not self.active:
            return
            
        current_time = self.sim.get_sim_time()
        
        # Check if we've reached the target time
        if current_time >= self.target_time:
            if self.waypoint_mode:
                # Try to advance to next waypoint
                self._advance_to_next_waypoint()
                if not self.active:
                    # Mission complete, hold position
                    self.sim['fcs/aileron-cmd-norm'] = 0.0
                    self.sim['fcs/elevator-cmd-norm'] = 0.0
                    # Keep altitude hold active
                    altitude_error = self.target_altitude - self.sim['position/h-sl-ft']
                    heave_adjustment = altitude_error * self.kp_z
                    heave_cmd = self.base_heave_cmd + heave_adjustment
                    heave_cmd = max(-0.5, min(1.0, heave_cmd))
                    self.sim['fcs/cmdHeave_nd'] = heave_cmd
                    return
            else:
                # Single waypoint mode - stop at target
                self.active = False
                self.sim['fcs/aileron-cmd-norm'] = 0.0
                self.sim['fcs/elevator-cmd-norm'] = 0.0
                # Keep altitude hold active
                altitude_error = self.target_altitude - self.sim['position/h-sl-ft']
                heave_adjustment = altitude_error * self.kp_z
                heave_cmd = self.base_heave_cmd + heave_adjustment
                heave_cmd = max(-0.5, min(1.0, heave_cmd))
                self.sim['fcs/cmdHeave_nd'] = heave_cmd
                print(f"Target time reached at {current_time:.1f}s")
                return
            
        # Get current position
        current_x = self.sim['position/distance-from-start-lon-mt'] * 3.28084
        current_y = self.sim['position/distance-from-start-lat-mt'] * 3.28084
        current_z = self.sim['position/h-sl-ft']
        
        # Calculate desired position based on trajectory
        desired_x, desired_y, desired_z = self.calculate_desired_position(current_time)
        
        # Calculate position errors
        error_x = desired_x - current_x
        error_y = desired_y - current_y
        error_z = desired_z - current_z
        
        # Get current velocities for velocity damping
        vel_x = self.sim['velocities/v-fps']  # East-West velocity
        vel_y = self.sim['velocities/u-fps']  # North-South velocity
        vel_z = self.sim['velocities/w-fps']  # Vertical velocity
        
        # Calculate desired velocities for feedforward control
        if self.target_time > self.start_time:
            desired_vel_x = (self.target_x - self.start_x) / (self.target_time - self.start_time)
            desired_vel_y = (self.target_y - self.start_y) / (self.target_time - self.start_time)
            desired_vel_z = (self.target_z - self.start_z) / (self.target_time - self.start_time)
        else:
            desired_vel_x = 0.0
            desired_vel_y = 0.0
            desired_vel_z = 0.0
        
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
        
        # PID control for z-direction (altitude, affects heave)
        self.error_z_integral += error_z * self.dt
        # Add integral windup protection
        self.error_z_integral = max(-50, min(50, self.error_z_integral))
        error_z_derivative = (error_z - self.prev_error_z) / self.dt
        
        # Enhanced heave command with feedforward and velocity damping
        heave_adjustment = (self.kp_z * error_z + 
                           self.ki_z * self.error_z_integral + 
                           self.kd_z * error_z_derivative +
                           self.ff_gain_z * desired_vel_z -
                           self.vel_damping_z * vel_z)
        
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
        
        # Apply heave command for altitude control
        heave_cmd = self.base_heave_cmd + heave_adjustment
        heave_cmd = max(-0.5, min(1.0, heave_cmd))  # Limit heave command
        self.sim['fcs/cmdHeave_nd'] = heave_cmd
        
        # Update previous errors for next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_z = error_z
        
        # Debug output every 2 seconds
        if int(current_time * 10) % 20 == 0:  # Every 2 seconds
            print(f"T={current_time:.1f}s: Pos=({current_x:.1f},{current_y:.1f},{current_z:.1f}), "
                  f"Desired=({desired_x:.1f},{desired_y:.1f},{desired_z:.1f}), "
                  f"Error=({error_x:.1f},{error_y:.1f},{error_z:.1f}), "
                  f"Cmd=(R:{roll_cmd:.3f},P:{pitch_cmd:.3f},H:{heave_cmd:.3f})")


def go_to_position(x, y, z, target_time):
    """Factory function to create a trajectory controller for a single waypoint.
    
    Args:
        x (float): Target x-position in feet (East-West direction)
        y (float): Target y-position in feet (North-South direction)
        z (float): Target z-position in feet (altitude)
        target_time (float): Time in seconds to reach the target position
        
    Returns:
        function: Factory function that creates a TrajectoryController when called with sim object
        
    Example:
        # Create a controller to go to position (100, 50, 320) at 15 seconds
        controller_factory = go_to_position(100, 50, 320, 15.0)
        controller = controller_factory(sim)
        
        # In the main simulation loop, call:
        # controller.update_control_commands()
    """
    def create_controller(sim):
        controller = TrajectoryController(sim)
        controller.set_target_position(x, y, z, target_time)
        return controller
    
    return create_controller


def go_to_waypoints(waypoints):
    """Factory function to create a trajectory controller for multiple waypoints.
    
    Args:
        waypoints: List of (x, y, z, time) tuples representing waypoints
        
    Returns:
        function: Factory function that creates a TrajectoryController when called with sim object
        
    Example:
        # Create a controller for multiple waypoints
        waypoints = [(50, 25, 320, 10), (100, 50, 300, 20), (0, 0, 320, 30)]
        controller_factory = go_to_waypoints(waypoints)
        controller = controller_factory(sim)
        
        # In the main simulation loop, call:
        # controller.update_control_commands()
    """
    def create_controller(sim):
        controller = TrajectoryController(sim)
        controller.set_waypoints(waypoints)
        return controller
    
    return create_controller
