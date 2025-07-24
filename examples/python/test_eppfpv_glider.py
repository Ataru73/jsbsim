#!/usr/bin/env python3
"""
EPPFPV Glider Test Script with Waypoint Navigation
=================================================

This script tests the EPPFPV aircraft in JSBSim as a glider with the following mission:
- Start at 1000 feet altitude and 25 knots airspeed
- Navigate through 4 waypoints while gliding
- Generate trajectory and altitude plots

The EPPFPV is treated as a glider with minimal engine power.
"""

import jsbsim
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

# Constants
DT = 0.0083333  # Simulation time step (120 Hz)

class GliderAutopilot:
    """Simple autopilot for glider waypoint navigation"""
    
    def __init__(self, sim):
        self.sim = sim
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_tolerance = 300.0  # feet (larger tolerance for glider)
        self.altitude_tolerance = 100.0   # feet (larger tolerance for glider)
        
        # Very gentle PID gains for glider control
        self.heading_kp = 0.005  # Very gentle heading control
        self.heading_ki = 0.0001
        self.heading_kd = 0.002
        self.heading_error_integral = 0.0
        self.prev_heading_error = 0.0
        
        # Minimal altitude control (gliders lose altitude naturally)
        self.altitude_kp = 0.002  # Very gentle altitude control
        self.altitude_ki = 0.00001
        self.altitude_kd = 0.001
        self.altitude_error_integral = 0.0
        self.prev_altitude_error = 0.0
        
        # Target speed for glider
        self.target_speed = 25.0  # knots (appropriate for glider)
        
    def set_waypoints(self, waypoints):
        """Set list of waypoints [(lat, lon, alt), ...]"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        print(f"Waypoints set: {len(waypoints)} waypoints")
        for i, (lat, lon, alt) in enumerate(waypoints):
            print(f"  WP{i+1}: Lat={lat:.6f}°, Lon={lon:.6f}°, Alt={alt:.0f}ft")
    
    def get_bearing_to_waypoint(self, current_lat, current_lon, target_lat, target_lon):
        """Calculate bearing from current position to target waypoint"""
        lat1 = math.radians(current_lat)
        lat2 = math.radians(target_lat)
        dlon = math.radians(target_lon - current_lon)
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        
        # Normalize to 0-360
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def get_distance_to_waypoint(self, current_lat, current_lon, target_lat, target_lon):
        """Calculate distance to waypoint in feet"""
        # Approximate distance calculation for small distances
        dlat = target_lat - current_lat
        dlon = target_lon - current_lon
        
        # Convert to feet (approximate)
        lat_ft = dlat * 364000  # feet per degree latitude
        lon_ft = dlon * 364000 * math.cos(math.radians(current_lat))  # feet per degree longitude
        
        distance = math.sqrt(lat_ft**2 + lon_ft**2)
        return distance
    
    def update(self):
        """Update autopilot controls for glider flight"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return False  # No more waypoints
        
        # Get current position and attitude
        current_lat = self.sim['position/lat-gc-deg']
        current_lon = self.sim['position/long-gc-deg']
        current_alt = self.sim['position/h-sl-ft']
        current_heading = self.sim['attitude/psi-deg']
        current_speed = self.sim['velocities/vc-kts']
        
        # Check for NaN values and handle gracefully
        if (math.isnan(current_lat) or math.isnan(current_lon) or 
            math.isnan(current_alt) or math.isnan(current_heading) or
            math.isnan(current_speed)):
            print(f"Warning: NaN values detected at time {self.sim.get_sim_time():.1f}s")
            return False
        
        # Get current waypoint
        target_lat, target_lon, target_alt = self.waypoints[self.current_waypoint_index]
        
        # Calculate bearing and distance to waypoint
        target_bearing = self.get_bearing_to_waypoint(current_lat, current_lon, target_lat, target_lon)
        distance = self.get_distance_to_waypoint(current_lat, current_lon, target_lat, target_lon)
        
        # Check if we've reached the current waypoint
        altitude_error = abs(current_alt - target_alt)
        if distance < self.waypoint_tolerance and altitude_error < self.altitude_tolerance:
            print(f"Reached waypoint {self.current_waypoint_index + 1} at time {self.sim.get_sim_time():.1f}s")
            print(f"  Position: Lat={current_lat:.6f}°, Lon={current_lon:.6f}°, Alt={current_alt:.0f}ft")
            print(f"  Distance error: {distance:.1f}ft, Altitude error: {altitude_error:.1f}ft")
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                print("All waypoints reached!")
                return False
            # Reset PID accumulators for new waypoint
            self.heading_error_integral = 0.0
            self.altitude_error_integral = 0.0
            return True
        
        # Heading control (very gentle for glider)
        heading_error = target_bearing - current_heading
        # Normalize heading error to [-180, 180]
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        
        self.heading_error_integral += heading_error * DT
        # Limit integral windup
        self.heading_error_integral = max(-10.0, min(10.0, self.heading_error_integral))
        heading_derivative = (heading_error - self.prev_heading_error) / DT
        
        # Try inverted rudder control (common JSBSim issue)
        rudder_cmd = -(self.heading_kp * heading_error + 
                      self.heading_ki * self.heading_error_integral + 
                      self.heading_kd * heading_derivative)
        rudder_cmd = max(-0.3, min(0.3, rudder_cmd))  # Limit rudder deflection
        
        self.prev_heading_error = heading_error
        
        # Altitude control (very gentle, accepting natural glide descent)
        altitude_error = target_alt - current_alt
        self.altitude_error_integral += altitude_error * DT
        # Limit integral windup
        self.altitude_error_integral = max(-100.0, min(100.0, self.altitude_error_integral))
        altitude_derivative = (altitude_error - self.prev_altitude_error) / DT
        
        elevator_cmd = (self.altitude_kp * altitude_error + 
                       self.altitude_ki * self.altitude_error_integral + 
                       self.altitude_kd * altitude_derivative)
        elevator_cmd = max(-0.2, min(0.2, elevator_cmd))  # Very limited elevator control
        
        self.prev_altitude_error = altitude_error
        
        # No throttle for glider mode
        throttle_cmd = 0.0
        
        # Apply controls with very gentle inputs
        self.sim['fcs/throttle-cmd-norm[0]'] = throttle_cmd
        self.sim['fcs/elevator-cmd-norm'] = elevator_cmd
        self.sim['fcs/rudder-cmd-norm'] = rudder_cmd
        self.sim['fcs/aileron-cmd-norm'] = 0.0  # Keep wings level
        
        # Debug output every 10 seconds
        if int(self.sim.get_sim_time()) % 10 == 0 and self.sim.get_sim_time() % 1.0 < DT:
            glide_ratio = current_speed / max(0.1, abs(self.sim['velocities/w-fps'] * 0.592484))  # Convert fps to kts
            print(f"Time: {self.sim.get_sim_time():.1f}s, WP{self.current_waypoint_index + 1}")
            print(f"  Current: Lat={current_lat:.6f}°, Lon={current_lon:.6f}°, Alt={current_alt:.0f}ft, Hdg={current_heading:.1f}°, Speed={current_speed:.1f}kts")
            print(f"  Target:  Lat={target_lat:.6f}°, Lon={target_lon:.6f}°, Alt={target_alt:.0f}ft, Bearing={target_bearing:.1f}°")
            print(f"  Errors:  Distance={distance:.1f}ft, Altitude={altitude_error:.1f}ft, Heading={heading_error:.1f}°")
            print(f"  Glide ratio: {glide_ratio:.1f}:1, Sink rate: {self.sim['velocities/w-fps']:.1f}fps")
            print(f"  Controls: Elevator={elevator_cmd:.3f}, Rudder={rudder_cmd:.3f}")
        
        return True

def test_eppfpv_glider():
    """Test EPPFPV aircraft in glider mode with waypoint navigation"""
    
    print("=== EPPFPV Glider Waypoint Navigation Test ===")
    print("Mission: Start at 1000ft, 25kts and glide through 2 waypoints")
    print("Note: EPPFPV operates as a glider with no engine power")
    print()
    
    # Initialize JSBSim
    sim = jsbsim.FGFDMExec('../..')
    
    # Load EPPFPV aircraft
    if not sim.load_model('EPPFPV'):
        print("Error: Could not load EPPFPV aircraft model")
        return
    
    print("EPPFPV aircraft loaded successfully")
    
    # Set simulation time step
    sim.set_dt(DT)
    
    # Define waypoints first to calculate initial heading
    initial_lat = 47.0  # Seattle area
    initial_lon = -122.0
    initial_alt = 1000.0  # feet MSL
    initial_speed = 25.0  # knots
    
    # Define 2 waypoints for realistic glider flight (closer, within 1000ft)
    # Each waypoint: (latitude, longitude, altitude)
    # Note: 0.001 degree lat ≈ 364 feet, 0.001 degree lon ≈ 230 feet at Seattle latitude
    waypoints = [
        (initial_lat + 0.0015, initial_lon + 0.0010, 800),   # WP1: ~600ft NE, descend to 800ft
        (initial_lat + 0.0025, initial_lon + 0.0020, 600),   # WP2: ~950ft NE, descend to 600ft
    ]
    
    # Calculate initial heading towards first waypoint
    target_lat, target_lon, _ = waypoints[0]
    lat1 = math.radians(initial_lat)
    lat2 = math.radians(target_lat)
    dlon = math.radians(target_lon - initial_lon)
    
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    initial_heading = math.degrees(math.atan2(y, x))
    initial_heading = (initial_heading + 360) % 360  # Normalize to 0-360
    
    # Set initial conditions with gentle trim for glider
    sim['ic/lat-gc-deg'] = initial_lat
    sim['ic/long-gc-deg'] = initial_lon
    sim['ic/h-sl-ft'] = initial_alt
    sim['ic/psi-true-deg'] = initial_heading
    sim['ic/vc-kts'] = initial_speed
    sim['ic/alpha-deg'] = 3.0  # Slightly positive AoA for glider
    sim['ic/beta-deg'] = 0.0
    sim['ic/phi-deg'] = 0.0  # No roll
    sim['ic/theta-deg'] = 0.0  # Level attitude initially
    sim['ic/p-rad_sec'] = 0.0
    sim['ic/q-rad_sec'] = 0.0
    sim['ic/r-rad_sec'] = 0.0
    
    # Initialize the simulation
    sim.run_ic()
    
    # Set engine to minimal power (glider mode)
    sim['fcs/throttle-cmd-norm[0]'] = 0.0  # Minimal throttle to avoid transients
    
    print(f"Initial conditions set:")
    print(f"  Position: Lat={initial_lat:.6f}°, Lon={initial_lon:.6f}°")
    print(f"  Altitude: {initial_alt:.0f}ft MSL")
    print(f"  Speed: {initial_speed:.0f}kts")
    print(f"  Heading: {initial_heading:.1f}° (pointing towards first waypoint)")
    print(f"  Mode: Glider (no engine power)")
    print()
    
    # Create autopilot and set waypoints
    autopilot = GliderAutopilot(sim)
    autopilot.set_waypoints(waypoints)
    
    # Data collection for plotting
    times = []
    latitudes = []
    longitudes = []
    altitudes = []
    speeds = []
    headings = []
    sink_rates = []
    
    # Convert lat/lon to relative X/Y coordinates for easier plotting
    positions_x = []  # East-West in feet
    positions_y = []  # North-South in feet
    
    # Run simulation
    print("Starting glider simulation...")
    max_time = 600.0  # Maximum simulation time (10 minutes)
    
    waypoints_reached = True
    simulation_stable = True
    
    while sim.get_sim_time() < max_time and waypoints_reached and simulation_stable:
        # Update autopilot
        waypoints_reached = autopilot.update()
        
        # Run one simulation step
        sim.run()
        
        # Collect data
        current_time = sim.get_sim_time()
        current_lat = sim['position/lat-gc-deg']
        current_lon = sim['position/long-gc-deg']
        current_alt = sim['position/h-sl-ft']
        current_speed = sim['velocities/vc-kts']
        current_heading = sim['attitude/psi-deg']
        current_sink_rate = sim['velocities/w-fps']  # Positive = descending
        
        # Check for simulation stability
        if (math.isnan(current_lat) or math.isnan(current_lon) or 
            math.isnan(current_alt) or current_alt < 200):  # Stop if too low
            print(f"Simulation ended at {current_time:.1f}s (altitude too low or unstable)")
            simulation_stable = False
            break
        
        times.append(current_time)
        latitudes.append(current_lat)
        longitudes.append(current_lon)
        altitudes.append(current_alt)
        speeds.append(current_speed)
        headings.append(current_heading)
        sink_rates.append(current_sink_rate)
        
        # Convert to relative coordinates (feet from starting position)
        dlat = current_lat - initial_lat
        dlon = current_lon - initial_lon
        pos_x = dlon * 364000 * math.cos(math.radians(initial_lat))  # East-West
        pos_y = dlat * 364000  # North-South
        positions_x.append(pos_x)
        positions_y.append(pos_y)
    
    print(f"\nSimulation completed at {sim.get_sim_time():.1f} seconds")
    if len(latitudes) > 0:
        print(f"Final position: Lat={latitudes[-1]:.6f}°, Lon={longitudes[-1]:.6f}°, Alt={altitudes[-1]:.0f}ft")
        
        # Create plots
        create_glider_plots(times, positions_x, positions_y, altitudes, speeds, headings, 
                           sink_rates, waypoints, initial_lat, initial_lon)
    else:
        print("No valid data collected - simulation failed immediately")
    
    return sim, times, positions_x, positions_y, altitudes

def create_glider_plots(times, positions_x, positions_y, altitudes, speeds, headings,
                       sink_rates, waypoints, initial_lat, initial_lon):
    """Create comprehensive plots of the glider flight trajectory"""
    
    # Convert waypoints to relative coordinates
    waypoint_x = []
    waypoint_y = []
    waypoint_alt = []
    
    for lat, lon, alt in waypoints:
        dlat = lat - initial_lat
        dlon = lon - initial_lon
        wp_x = dlon * 364000 * math.cos(math.radians(initial_lat))
        wp_y = dlat * 364000
        waypoint_x.append(wp_x)
        waypoint_y.append(wp_y)
        waypoint_alt.append(alt)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Ground track (X-Y trajectory)
    if len(positions_x) > 0:
        ax1.plot(positions_x, positions_y, 'b-', linewidth=2, label='Glider Path')
        ax1.scatter(waypoint_x, waypoint_y, c='red', s=100, marker='o', label='Waypoints', zorder=5)
        ax1.scatter([0], [0], c='green', s=150, marker='^', label='Start', zorder=5)
        
        # Add waypoint labels
        for i, (x, y, alt) in enumerate(zip(waypoint_x, waypoint_y, waypoint_alt)):
            ax1.annotate(f'WP{i+1}\n{alt:.0f}ft', (x, y), xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8)
    
    ax1.set_xlabel('East-West Position (feet)')
    ax1.set_ylabel('North-South Position (feet)')
    ax1.set_title('EPPFPV Glider Ground Track')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Altitude vs Time
    if len(times) > 0:
        ax2.plot(times, altitudes, 'b-', linewidth=2, label='Glider Altitude')
        
        # Add horizontal lines for waypoint altitudes
        for i, alt in enumerate(waypoint_alt):
            ax2.axhline(y=alt, color='red', linestyle='--', alpha=0.5, 
                       label=f'WP{i+1} Target' if i == 0 else '')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Altitude (feet MSL)')
    ax2.set_title('Glider Altitude Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed and Sink Rate vs Time
    if len(times) > 0:
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(times, speeds, 'g-', linewidth=2, label='Airspeed (kts)')
        line2 = ax3_twin.plot(times, sink_rates, 'r-', linewidth=2, label='Sink Rate (fps)')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Airspeed (knots)', color='g')
        ax3_twin.set_ylabel('Sink Rate (fps)', color='r')
        ax3.set_title('Glider Performance')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heading vs Time
    if len(times) > 0:
        ax4.plot(times, headings, 'm-', linewidth=2, label='Heading')
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Heading (degrees)')
    ax4.set_title('Glider Heading Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 360)
    
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle('EPPFPV Glider - 2 Waypoint Mission Analysis', fontsize=16, y=0.98)
    
    # Save the plot
    plt.savefig('eppfpv_glider_mission.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'eppfpv_glider_mission.png'")
    
    # Show plot
    plt.show()
    
    # Print mission statistics
    if len(times) > 0:
        total_distance = calculate_total_distance(positions_x, positions_y)
        altitude_lost = altitudes[0] - altitudes[-1] if len(altitudes) > 1 else 0
        avg_sink_rate = np.mean(sink_rates) if len(sink_rates) > 0 else 0
        glide_ratio = total_distance / max(1, altitude_lost) if altitude_lost > 0 else 0
        
        print("\nGlider Mission Statistics:")
        print(f"  Total flight time: {times[-1]:.1f} seconds ({times[-1]/60:.1f} minutes)")
        print(f"  Total distance: {total_distance:.1f} feet ({total_distance/5280:.2f} miles)")
        print(f"  Altitude lost: {altitude_lost:.0f} feet")
        print(f"  Average sink rate: {avg_sink_rate:.1f} fps")
        print(f"  Average speed: {np.mean(speeds):.1f} knots")
        print(f"  Overall glide ratio: {glide_ratio/5280:.1f}:1 (distance in miles per 1000ft altitude)")
        print(f"  Final altitude: {altitudes[-1]:.0f} feet")

def calculate_total_distance(x_coords, y_coords):
    """Calculate total distance traveled"""
    total_distance = 0.0
    if len(x_coords) > 1:
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_distance += math.sqrt(dx**2 + dy**2)
    return total_distance

if __name__ == "__main__":
    try:
        test_eppfpv_glider()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure you're running this script from the JSBSim examples/python directory")
        print("2. Verify that the EPPFPV aircraft model exists in the aircraft directory")
        print("3. Check that JSBSim Python bindings are properly installed")
        print("4. Ensure matplotlib is installed for plotting: pip install matplotlib")
