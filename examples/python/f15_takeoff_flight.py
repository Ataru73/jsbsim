#!/usr/bin/env python3
"""
F15 Takeoff and Steady Flight Simulation
========================================

This script simulates the complete takeoff sequence and transition to steady flight
for an F15 Eagle using JSBSim. The simulation includes:

1. Engine startup: Both afterburning turbofan engines
2. Takeoff roll: Full afterburner, rotation at appropriate speed
3. Initial climb: Rapid climb to pattern altitude
4. Cruise climb: Transition to cruise altitude
5. Level flight: Steady state cruise flight

Aircraft: F15 Eagle
Engines: 2x Pratt & Whitney F100-PW-100 turbofans
Typical takeoff speed: 150-180 kts
Max thrust: ~25,000 lbf per engine with afterburner
"""

import jsbsim
import matplotlib.pyplot as plt
import numpy as np
import math

# Configuration
AIRCRAFT_NAME = "f15"
PATH_TO_JSBSIM_FILES = "../.."
# DT = 0.0083333
DT = 1 / 20.0  # 20 Hz update rate

# Define a list of waypoints to follow
# Each waypoint is a dictionary: {'lat': latitude, 'lon': longitude, 'alt': altitude_ft, 'speed': speed_kts}
WAYPOINTS = [
    {'lat': 41.70, 'lon': 12.45, 'alt': 5000, 'speed': 300},
    {'lat': 41.75, 'lon': 12.55, 'alt': 6000, 'speed': 320},
    {'lat': 41.70, 'lon': 12.65, 'alt': 5000, 'speed': 300},
    {'lat': 41.65, 'lon': 12.55, 'alt': 4000, 'speed': 280},
]

# Define obstacles (optional)
# Each obstacle is a cylindrical obstacle with infinite height
# Dictionary format: {'lat': latitude, 'lon': longitude, 'radius_m': radius_in_meters}
EXAMPLE_OBSTACLES = [
    {'lat': 41.725, 'lon': 12.500, 'radius_m': 400},  # Obstacle directly between WP0 and WP1
    {'lat': 41.675, 'lon': 12.600, 'radius_m': 350},  # Obstacle directly between WP2 and WP3
]

def initialize_f15():
    """Initialize JSBSim with F15"""
    jsbsim.FGJSBBase().debug_lvl = 0
    fdm = jsbsim.FGFDMExec(PATH_TO_JSBSIM_FILES)
    fdm.load_model(AIRCRAFT_NAME)
    fdm.set_dt(DT)
    
    # Set initial position near the first waypoint
    fdm['ic/lat-gc-deg'] = 41.658
    fdm['ic/long-gc-deg'] = 12.446
    fdm['ic/terrain-elevation-ft'] = 160
    
    fdm['ic/h-agl-ft'] = 5
    fdm['ic/psi-true-deg'] = 180.0
    fdm['ic/vc-kts'] = 0.0
    
    # Set brakes on initially
    fdm['fcs/left-brake-cmd-norm'] = 0.9
    fdm['fcs/right-brake-cmd-norm'] = 0.9
    
    fdm.run_ic()
    
    print(f"Aircraft initialized: {AIRCRAFT_NAME}")
    return fdm

class F15AutoPilot:
    def __init__(self, fdm, waypoints, obstacles=None):
        self.fdm = fdm
        self.waypoints = waypoints
        self.obstacles = obstacles or []  # List of obstacles: {'lat': lat, 'lon': lon, 'radius_m': radius}
        self.current_waypoint_idx = 0
        self.engines_started = False
        self.takeoff_roll_started = False
        self.airborne = False
        self.gear_retracted = False
        self.altitude_pid_integral = 0.0
        self.filtered_roll_rate_dps = 0.0
        self.roll_integral = 0.0
        self.waypoint_tolerance_m = 250  # Distance to waypoint to consider it "reached"
        self.obstacle_avoidance_distance_m = 500  # Start avoiding obstacles at this distance
        self.obstacle_safe_altitude_ft = 1000  # Minimum altitude to clear obstacles
        
        # Obstacle avoidance state tracking
        self.avoiding_obstacle = False
        self.avoidance_start_time = 0
        self.min_avoidance_time = 30  # Minimum time to maintain avoidance (seconds)
        self.avoided_obstacles = set()  # Track obstacles we've already avoided
        
    def execute_takeoff_sequence(self, sim_time):
        if sim_time >= 0.1:
            self.fdm['fcs/flap-cmd-norm'] = 0.3
        if sim_time >= 0.2:
            self.fdm['fcs/left-brake-cmd-norm'] = 0.0
            self.fdm['fcs/right-brake-cmd-norm'] = 0.0
        if sim_time >= 0.3:
            self.fdm['propulsion/starter_cmd'] = 1 
        if sim_time >= 0.4 and not self.engines_started:
            self.fdm['propulsion/cutoff_cmd'] = 0
            self.engines_started = True
            print(f"[{sim_time:5.1f}s] Engines started.")
        if sim_time >= 1.0 and not self.takeoff_roll_started:
            self.fdm['fcs/throttle-cmd-norm[0]'] = 1.0
            self.fdm['fcs/throttle-cmd-norm[1]'] = 1.0
            self.takeoff_roll_started = True
            print(f"[{sim_time:5.1f}s] Full throttle - takeoff roll.")
        if self.takeoff_roll_started and not self.airborne and self.fdm['velocities/vc-kts'] > 120:
            self.fdm['fcs/elevator-cmd-norm'] = -0.4
        if self.takeoff_roll_started and not self.airborne and self.fdm['position/h-agl-ft'] > 20:
            self.airborne = True
            print(f"[{sim_time:5.1f}s] Airborne.")
        if self.airborne and sim_time > 45.0 and not self.gear_retracted:
            self.fdm['fcs/flap-cmd-norm'] = 0.0
            self.fdm['gear/gear-cmd-norm'] = 0
            self.gear_retracted = True
            print(f"[{sim_time:5.1f}s] Gear and flaps retracted.")

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dLon = lon2_rad - lon1_rad
        y = math.sin(dLon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dLon)
        bearing_rad = math.atan2(y, x)
        return (math.degrees(bearing_rad) + 360) % 360

    def calculate_distance_m(self, lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
        
    def detect_obstacles(self, current_lat, current_lon, heading_deg, lookahead_distance_m=3000, debug=False):
        """Detect obstacles in the current flight path - SIMPLIFIED VERSION
        
        Args:
            current_lat: Current aircraft latitude
            current_lon: Current aircraft longitude 
            heading_deg: Current aircraft heading in degrees
            lookahead_distance_m: Distance to look ahead for obstacles
            debug: Print debug information
            
        Returns:
            List of threatening obstacles with their distances and relative bearings
        """
        threatening_obstacles = []
        
        for i, obstacle in enumerate(self.obstacles):
            # Calculate distance to obstacle
            distance_to_obstacle = self.calculate_distance_m(
                current_lat, current_lon, 
                obstacle['lat'], obstacle['lon']
            )
            
            # Calculate bearing to obstacle
            bearing_to_obstacle = self.calculate_bearing(
                current_lat, current_lon,
                obstacle['lat'], obstacle['lon']
            )
            
            # Calculate relative bearing (obstacle bearing relative to aircraft heading)
            relative_bearing = (bearing_to_obstacle - heading_deg + 180) % 360 - 180
            
            # Debug output for obstacle detection
            if debug:
                print(f"  Obstacle {i}: dist={distance_to_obstacle:.0f}m, bearing={bearing_to_obstacle:.1f}°, rel_bearing={relative_bearing:.1f}°")
            
            # SIMPLIFIED THREAT DETECTION - Much more aggressive
            is_threatening = False
            threat_reason = ""
            
            # Rule 1: Any obstacle within 1500m is a threat
            if distance_to_obstacle <= 1500:
                is_threatening = True
                threat_reason = f"close proximity ({distance_to_obstacle:.0f}m < 1500m)"
            
            # Rule 2: Any obstacle in forward hemisphere (±90°) within 2500m is a threat
            elif abs(relative_bearing) <= 90 and distance_to_obstacle <= 2500:
                is_threatening = True
                threat_reason = f"forward hemisphere ({distance_to_obstacle:.0f}m < 2500m, {relative_bearing:.1f}°)"
            
            # Rule 3: Any obstacle directly ahead (±45°) within 3000m is a threat
            elif abs(relative_bearing) <= 45 and distance_to_obstacle <= lookahead_distance_m:
                is_threatening = True
                threat_reason = f"direct path ({distance_to_obstacle:.0f}m < 3000m, {relative_bearing:.1f}°)"
            
            if is_threatening:
                if debug:
                    print(f"    -> THREAT! {threat_reason}")
                threatening_obstacles.append({
                    'obstacle': obstacle,
                    'distance_m': distance_to_obstacle,
                    'relative_bearing_deg': relative_bearing,
                    'threat_distance_m': distance_to_obstacle,  # Use actual distance
                    'threat_reason': threat_reason
                })
        
        # Sort by distance (closest first)
        threatening_obstacles.sort(key=lambda x: x['distance_m'])
        return threatening_obstacles
        
    def calculate_avoidance_maneuver(self, threatening_obstacles, current_altitude, sim_time):
        """Calculate avoidance maneuver for detected obstacles - IMPROVED VERSION
        
        Args:
            threatening_obstacles: List of threatening obstacles from detect_obstacles()
            current_altitude: Current aircraft altitude
            sim_time: Current simulation time
            
        Returns:
            Dictionary with avoidance commands: {'altitude_adjust': ft, 'heading_adjust': deg, 'is_avoiding': bool}
        """
        # Check if we should continue avoiding (minimum avoidance time)
        if self.avoiding_obstacle and (sim_time - self.avoidance_start_time) < self.min_avoidance_time:
            # Continue previous avoidance maneuver but with reduced intensity over time
            time_remaining = self.min_avoidance_time - (sim_time - self.avoidance_start_time)
            
            # Gradual reduction in avoidance intensity
            intensity_factor = max(0.3, time_remaining / self.min_avoidance_time)  # From 1.0 to 0.3
            
            altitude_adjust = max(current_altitude + (1000 * intensity_factor), self.obstacle_safe_altitude_ft + 500)
            heading_adjust = 30 * intensity_factor  # Reduce turn angle over time
            
            if time_remaining > 15:
                print(f"[ACTIVE AVOIDANCE] Time remaining: {time_remaining:.1f}s, intensity: {intensity_factor:.2f}")
            else:
                print(f"[ENDING AVOIDANCE] Time remaining: {time_remaining:.1f}s, intensity: {intensity_factor:.2f}")
                
            return {'altitude_adjust': altitude_adjust, 'heading_adjust': heading_adjust, 'is_avoiding': True}
        
        # If no new threats, stop avoiding
        if not threatening_obstacles:
            if self.avoiding_obstacle:
                print(f"[AVOIDANCE COMPLETE] Resuming waypoint navigation")
                self.avoiding_obstacle = False
            return {'altitude_adjust': 0, 'heading_adjust': 0, 'is_avoiding': False}
            
        # New threat detected - start/continue avoidance
        primary_threat = threatening_obstacles[0]
        relative_bearing = primary_threat['relative_bearing_deg']
        distance = primary_threat['distance_m']
        
        # Start new avoidance if not already avoiding
        if not self.avoiding_obstacle:
            self.avoiding_obstacle = True
            self.avoidance_start_time = sim_time
            print(f"\n*** NEW OBSTACLE AVOIDANCE STARTED ***")
        else:
            print(f"\n*** CONTINUING OBSTACLE AVOIDANCE ***")
            
        print(f"Threat: {primary_threat['threat_reason']}")
        print(f"Distance: {distance:.0f}m, Relative bearing: {relative_bearing:.1f}°")
        
        # More moderate altitude avoidance
        altitude_adjust = max(current_altitude + 1500, self.obstacle_safe_altitude_ft + 1000)
        print(f"ALTITUDE AVOIDANCE: Climbing to {altitude_adjust:.0f} ft")
        
        # Moderate horizontal avoidance - avoid excessive turns
        base_turn_angle = 45  # Smaller base turn angle
        
        # Turn away from obstacle (opposite direction of relative bearing)
        if relative_bearing > 0:  # Obstacle to the right
            heading_adjust = -base_turn_angle  # Turn left
        else:  # Obstacle to the left  
            heading_adjust = base_turn_angle   # Turn right
            
        # Scale turn based on distance but cap it
        if distance < 1500:
            distance_factor = min(2.0, 1500 / max(distance, 100))  # Cap the multiplier
            heading_adjust *= distance_factor
            heading_adjust = max(-75, min(75, heading_adjust))  # Smaller maximum turns
        
        print(f"HORIZONTAL AVOIDANCE: Turning {heading_adjust:.1f} degrees")
        print(f"Avoidance will continue for at least {self.min_avoidance_time:.1f} seconds")
        print(f"*** END OBSTACLE AVOIDANCE ***\n")
        
        return {'altitude_adjust': altitude_adjust, 'heading_adjust': heading_adjust, 'is_avoiding': True}

    def execute_in_flight_control(self, sim_time):
        if self.airborne:
            lat, lon = self.fdm['position/lat-gc-deg'], self.fdm['position/long-gc-deg']
            current_altitude = self.fdm['position/h-agl-ft']
            heading = self.fdm['attitude/psi-deg']
            
            # Obstacle Detection and Avoidance (check every 5 seconds for debug output)
            if int(sim_time) % 5 == 0 and abs(sim_time - int(sim_time)) < 0.1:
                print(f"[{sim_time:5.1f}s] Obstacle check at Lat:{lat:.4f}, Lon:{lon:.4f}, Heading:{heading:.1f}°, Alt:{current_altitude:.0f}ft")
            threatening_obstacles = self.detect_obstacles(lat, lon, heading, debug=(int(sim_time) % 5 == 0 and abs(sim_time - int(sim_time)) < 0.1))
            avoidance_maneuver = self.calculate_avoidance_maneuver(threatening_obstacles, current_altitude, sim_time)
            
            # Waypoint Navigation Logic
            if self.current_waypoint_idx < len(self.waypoints):
                current_wp = self.waypoints[self.current_waypoint_idx]
                dist_to_wp = self.calculate_distance_m(lat, lon, current_wp['lat'], current_wp['lon'])

                if dist_to_wp < self.waypoint_tolerance_m:  # Waypoint capture distance in meters
                    print(f"[{sim_time:5.1f}s] Reached Waypoint {self.current_waypoint_idx}. Distance: {dist_to_wp:.2f}m")
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx >= len(self.waypoints):
                        print(f"[{sim_time:5.1f}s] Final waypoint reached. Maintaining course.")
                    else:
                        next_wp = self.waypoints[self.current_waypoint_idx]
                        print(f"[{sim_time:5.1f}s] Heading to Waypoint {self.current_waypoint_idx} at {next_wp['alt']} ft, {next_wp['speed']} kts.")

            # Determine target parameters with improved mode switching
            if self.current_waypoint_idx < len(self.waypoints):
                # We have active waypoints to navigate to
                target_wp = self.waypoints[self.current_waypoint_idx]
                
                if avoidance_maneuver['is_avoiding']:
                    # AVOIDANCE MODE: Override waypoint navigation
                    if avoidance_maneuver['heading_adjust'] != 0:
                        # Turn away from obstacle (relative to current heading)
                        target_bearing = (heading + avoidance_maneuver['heading_adjust']) % 360
                    else:
                        # No turn needed, maintain current heading
                        target_bearing = heading
                        
                    # Set altitude for obstacle avoidance
                    target_altitude = avoidance_maneuver['altitude_adjust']
                    target_airspeed = max(target_wp['speed'], 300)  # Maintain higher speed during avoidance
                    
                    print(f"[AVOIDANCE MODE] Target bearing: {target_bearing:.1f}°, Target alt: {target_altitude:.0f}ft")
                    
                else:
                    # WAYPOINT MODE: Navigate normally to waypoint
                    target_bearing = self.calculate_bearing(lat, lon, target_wp['lat'], target_wp['lon'])
                    target_altitude = target_wp['alt']
                    target_airspeed = target_wp['speed']
                    
                    # Debug output for waypoint navigation
                    dist_to_wp = self.calculate_distance_m(lat, lon, target_wp['lat'], target_wp['lon'])
                    if int(sim_time) % 10 == 0 and abs(sim_time - int(sim_time)) < 0.1:
                        print(f"[WAYPOINT MODE] Target WP{self.current_waypoint_idx}: {target_bearing:.1f}°, {dist_to_wp:.0f}m")
                
                # Calculate heading error and roll command
                heading_error = (target_bearing - heading + 180) % 360 - 180
                target_roll_deg = max(-45, min(45, heading_error * 1.5))
                
            else:
                # All waypoints completed
                last_wp = self.waypoints[-1]
                
                if avoidance_maneuver['is_avoiding']:
                    # Still need to avoid obstacles even after waypoints complete
                    if avoidance_maneuver['heading_adjust'] != 0:
                        target_bearing = (heading + avoidance_maneuver['heading_adjust']) % 360
                        heading_error = (target_bearing - heading + 180) % 360 - 180
                        target_roll_deg = max(-45, min(45, heading_error * 1.5))
                    else:
                        target_roll_deg = 0  # Level wings
                        
                    target_altitude = avoidance_maneuver['altitude_adjust']
                    target_airspeed = max(last_wp['speed'], 300)
                else:
                    # Maintain level flight at last waypoint parameters
                    target_roll_deg = 0  # Level wings
                    target_altitude = last_wp['alt']
                    target_airspeed = last_wp['speed']

            # Roll control
            roll_angle_deg = self.fdm['attitude/phi-deg']
            roll_error = target_roll_deg - roll_angle_deg
            self.roll_integral += roll_error * DT
            self.roll_integral = max(-15.0, min(15.0, self.roll_integral))
            raw_roll_rate_dps = self.fdm['velocities/p-rad_sec'] * 57.2958
            alpha = 0.2
            self.filtered_roll_rate_dps = (alpha * raw_roll_rate_dps) + ((1 - alpha) * self.filtered_roll_rate_dps)
            kp_roll, ki_roll, kd_roll = 0.02, 0.004, 0.15
            aileron_cmd = (kp_roll * roll_error) + (ki_roll * self.roll_integral) + (kd_roll * -self.filtered_roll_rate_dps)
            self.fdm['fcs/aileron-cmd-norm'] = max(-0.25, min(0.25, aileron_cmd))

            # Altitude and Speed control
            altitude_error = target_altitude - self.fdm['position/h-agl-ft']
            altitude_tolerance = 10.0  # ft, adjust as needed

            if abs(altitude_error) > altitude_tolerance:
                self.altitude_pid_integral += altitude_error * DT
                self.altitude_pid_integral = max(-2000, min(2000, self.altitude_pid_integral))
                kp_alt, ki_alt, kd_alt = 0.0007, 0.00001, -0.015
                raw_elevator_cmd = -kp_alt * altitude_error - ki_alt * self.altitude_pid_integral + kd_alt * (-self.fdm['velocities/h-dot-fps'])
            else:
                raw_elevator_cmd = self.fdm['fcs/elevator-cmd-norm']  # No correction within tolerance

            # Apply low-pass filter to elevator command
            if not hasattr(self, 'filtered_elevator_cmd'):
                self.filtered_elevator_cmd = raw_elevator_cmd
            alpha_elev = 0.05  # Smoothing factor (0 < alpha < 1)
            self.filtered_elevator_cmd = alpha_elev * raw_elevator_cmd + (1 - alpha_elev) * self.filtered_elevator_cmd

            self.fdm['fcs/elevator-cmd-norm'] = max(-0.6, min(0.3, self.filtered_elevator_cmd))
            
            airspeed_error = target_airspeed - self.fdm['velocities/vc-kts']
            throttle_cmd = 1.0 if altitude_error > 500 or airspeed_error > 25 else 0.75
            self.fdm['fcs/throttle-cmd-norm[0]'] = throttle_cmd
            self.fdm['fcs/throttle-cmd-norm[1]'] = throttle_cmd

def run_simulation(obstacles=None):
    """Run the F15 takeoff and flight simulation"""
    fdm = initialize_f15()
    autopilot = F15AutoPilot(fdm, WAYPOINTS, obstacles)

    data = {
        'time': [], 'altitude': [], 'airspeed': [], 'roll': [], 'aileron': [],
        'lat': [], 'lon': [], 'elevator': [], 'throttle_left': [], 'throttle_right': [],
        'rudder': [], 'flap': [], 'gear': [], 'g_force': [], 'g_lateral': [], 'g_vertical': [],
        'fcs/aileron-pos-norm': [], 'fcs/elevator-pos-norm': [], 'fcs/rudder-pos-norm': []
    }
    simulation_time = 1200.0
    
    print(f"\nStarting F15 simulation...")
    
    while fdm.get_sim_time() < simulation_time:
        sim_time = fdm.get_sim_time()
        autopilot.execute_takeoff_sequence(sim_time)
        autopilot.execute_in_flight_control(sim_time)

        fdm.run()
        
        # Get accelerometer data (in ft/s^2, convert to G's by dividing with 32.174)
        ax_fps2 = fdm['accelerations/a-pilot-x-ft_sec2']  # Longitudinal (forward/back)
        ay_fps2 = fdm['accelerations/a-pilot-y-ft_sec2']  # Lateral (left/right)
        az_fps2 = fdm['accelerations/a-pilot-z-ft_sec2']  # Vertical (up/down)
        
        # Convert to G forces (1 G = 32.174 ft/s^2)
        g_longitudinal = ax_fps2 / 32.174
        g_lateral = ay_fps2 / 32.174
        g_vertical = -az_fps2 / 32.174  # Negative because JSBSim Z is positive down
        
        # Calculate total G force magnitude
        g_total = math.sqrt(g_longitudinal**2 + g_lateral**2 + g_vertical**2)
        
        data['time'].append(sim_time)
        data['altitude'].append(fdm['position/h-agl-ft'])
        data['airspeed'].append(fdm['velocities/vc-kts'])
        data['roll'].append(fdm['attitude/phi-deg'])
        data['aileron'].append(fdm['fcs/aileron-cmd-norm'])
        data['lat'].append(fdm['position/lat-gc-deg'])
        data['lon'].append(fdm['position/long-gc-deg'])
        data['elevator'].append(fdm['fcs/elevator-cmd-norm'])
        data['throttle_left'].append(fdm['fcs/throttle-cmd-norm[0]'])
        data['throttle_right'].append(fdm['fcs/throttle-cmd-norm[1]'])
        data['rudder'].append(fdm['fcs/rudder-cmd-norm'])
        data['flap'].append(fdm['fcs/flap-cmd-norm'])
        data['gear'].append(fdm['gear/gear-cmd-norm'])
        data['g_force'].append(g_total)
        data['g_lateral'].append(g_lateral)
        data['g_vertical'].append(g_vertical)
        # Collect actual control surface positions
        data['fcs/aileron-pos-norm'].append(fdm['fcs/aileron-pos-norm'])
        data['fcs/elevator-pos-norm'].append(fdm['fcs/elevator-pos-norm'])
        data['fcs/rudder-pos-norm'].append(fdm['fcs/rudder-pos-norm'])
    
    print("\nSimulation completed!")
    return data

def create_plots(data, obstacles=None):
    """Create plots of the key flight parameters."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('F15 Eagle - Waypoint Navigation Simulation', fontsize=16, fontweight='bold')
    times = data['time']

    # Altitude
    ax1.plot(times, data['altitude'], 'b-', linewidth=2)
    ax1.set_title('Altitude Profile')
    ax1.set_ylabel('Altitude (ft AGL)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(0, 10000)

    # Airspeed
    ax2.plot(times, data['airspeed'], 'r-', linewidth=2)
    ax2.set_title('Airspeed Profile')
    ax2.set_ylabel('Speed (kts)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Roll Angle
    ax3.plot(times, data['roll'], 'g-', linewidth=2)
    ax3.set_title('Roll Attitude')
    ax3.set_ylabel('Roll Angle (degrees)')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Ground Track
    ax4.plot(data['lon'], data['lat'], 'purple', linewidth=2, label='Flight Path')
    wp_lons = [wp['lon'] for wp in WAYPOINTS]
    wp_lats = [wp['lat'] for wp in WAYPOINTS]
    ax4.scatter(wp_lons, wp_lats, c='red', marker='x', s=100, label='Waypoints')
    for i, wp in enumerate(WAYPOINTS):
        ax4.text(wp['lon'], wp['lat'], f"  WP{i}", fontsize=9)
    
    # Plot obstacles if provided
    if obstacles:
        for i, obstacle in enumerate(obstacles):
            # Convert radius from meters to approximate degrees for visualization
            # This is approximate and assumes roughly 111,000 meters per degree
            radius_deg = obstacle['radius_m'] / 111000.0
            circle = plt.Circle((obstacle['lon'], obstacle['lat']), radius_deg, 
                              fill=False, color='orange', linewidth=2, linestyle='--')
            ax4.add_patch(circle)
            ax4.scatter([obstacle['lon']], [obstacle['lat']], c='orange', marker='o', s=50)
            ax4.text(obstacle['lon'], obstacle['lat'], f"  Obs{i}\n  {obstacle['radius_m']}m", fontsize=8, color='orange')
        ax4.legend(loc='best')
    else:
        ax4.legend()
        
    ax4.set_title('Ground Track')
    ax4.set_ylabel('Latitude (deg)')
    ax4.set_xlabel('Longitude (deg)')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_takeoff_flight_simulation.png', dpi=300)
    plt.show()

def create_command_plots(data):
    """Create plots of the command values (control surface and throttle commands)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('F15 Eagle - Flight Command Values', fontsize=16, fontweight='bold')
    times = data['time']

    # Control Surface Commands
    ax1.plot(times, data['aileron'], 'b-', linewidth=2, label='Aileron Command')
    ax1.plot(times, data['elevator'], 'r-', linewidth=2, label='Elevator Command')
    ax1.plot(times, data['rudder'], 'g-', linewidth=2, label='Rudder Command')
    ax1.set_title('Control Surface Commands')
    ax1.set_ylabel('Command (normalized)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(-1.0, 1.0)

    # Throttle Commands
    ax2.plot(times, data['throttle_left'], 'orange', linewidth=2, label='Left Engine')
    ax2.plot(times, data['throttle_right'], 'red', linewidth=2, label='Right Engine')
    ax2.set_title('Throttle Commands')
    ax2.set_ylabel('Throttle (normalized)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.1)

    # Configuration Commands
    ax3.plot(times, data['flap'], 'purple', linewidth=2, label='Flaps')
    ax3.plot(times, data['gear'], 'brown', linewidth=2, label='Landing Gear')
    ax3.set_title('Configuration Commands')
    ax3.set_ylabel('Command (normalized)')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(-0.1, 1.1)

    # Combined Control Overview
    ax4.plot(times, data['aileron'], 'b-', linewidth=1, alpha=0.7, label='Aileron')
    ax4.plot(times, data['elevator'], 'r-', linewidth=1, alpha=0.7, label='Elevator')
    ax4.plot(times, [t * 0.5 for t in data['throttle_left']], 'orange', linewidth=2, label='Throttle (scaled)')
    ax4.set_title('Control Commands Overview')
    ax4.set_ylabel('Command (normalized)')
    ax4.set_xlabel('Time (seconds)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_ylim(-0.6, 0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_command_values.png', dpi=300)
    plt.show()

def create_control_surface_position_plots(data):
    """Create plots of the actual control surface positions over time in a separate figure."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('F15 Eagle - Control Surface Positions', fontsize=16, fontweight='bold')
    times = data['time']

    # Aileron Position
    if 'fcs/aileron-pos-norm' in data:
        ax1.plot(times, data['fcs/aileron-pos-norm'], 'b-', linewidth=2, label='Aileron Position')
    else:
        ax1.plot(times, data['aileron'], 'b--', linewidth=1, label='Aileron Command (no position data)')
    ax1.set_title('Aileron Position')
    ax1.set_ylabel('Normalized Position')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(-1.0, 1.0)

    # Elevator Position
    if 'fcs/elevator-pos-norm' in data:
        ax2.plot(times, data['fcs/elevator-pos-norm'], 'r-', linewidth=2, label='Elevator Position')
    else:
        ax2.plot(times, data['elevator'], 'r--', linewidth=1, label='Elevator Command (no position data)')
    ax2.set_title('Elevator Position')
    ax2.set_ylabel('Normalized Position')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(-1.0, 1.0)

    # Rudder Position
    if 'fcs/rudder-pos-norm' in data:
        ax3.plot(times, data['fcs/rudder-pos-norm'], 'g-', linewidth=2, label='Rudder Position')
    else:
        ax3.plot(times, data['rudder'], 'g--', linewidth=1, label='Rudder Command (no position data)')
    ax3.set_title('Rudder Position')
    ax3.set_ylabel('Normalized Position')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(-1.0, 1.0)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_control_surface_positions.png', dpi=300)
    plt.show()

def create_g_force_plots(data):
    """Create plots of G force acceleration data."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('F15 Eagle - G Force Analysis', fontsize=16, fontweight='bold')
    times = data['time']

    # Total G Force
    ax1.plot(times, data['g_force'], 'r-', linewidth=2, label='Total G Force')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='1G Reference')
    ax1.axhline(y=6.0, color='orange', linestyle='--', alpha=0.7, label='6G Limit (typical)')
    ax1.axhline(y=9.0, color='red', linestyle='--', alpha=0.7, label='9G Max (F15)')
    ax1.set_title('Total G Force Magnitude')
    ax1.set_ylabel('G Force (G\'s)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(0, max(10, max(data['g_force']) + 1))

    # Vertical G Force (most important for pilot)
    ax2.plot(times, data['g_vertical'], 'b-', linewidth=2, label='Vertical G (up/down)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='1G Reference')
    ax2.axhline(y=0.0, color='black', linestyle='-', alpha=0.3, label='Zero G')
    ax2.axhline(y=-3.0, color='orange', linestyle='--', alpha=0.7, label='-3G Negative Limit')
    ax2.set_title('Vertical G Force (Pilot Experience)')
    ax2.set_ylabel('G Force (G\'s)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Lateral G Force
    ax3.plot(times, data['g_lateral'], 'g-', linewidth=2, label='Lateral G (left/right)')
    ax3.axhline(y=0.0, color='black', linestyle='-', alpha=0.3, label='Zero G')
    ax3.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='+4G Limit')
    ax3.axhline(y=-4.0, color='orange', linestyle='--', alpha=0.7, label='-4G Limit')
    ax3.set_title('Lateral G Force (Turning Forces)')
    ax3.set_ylabel('G Force (G\'s)')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    # G Force vs Roll Angle (to show correlation with maneuvers)
    ax4.scatter(data['roll'], data['g_force'], c=times, cmap='viridis', alpha=0.6, s=10)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='1G Reference')
    ax4.set_title('G Force vs Roll Angle (colored by time)')
    ax4.set_xlabel('Roll Angle (degrees)')
    ax4.set_ylabel('Total G Force (G\'s)')
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar for time
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Time (seconds)', rotation=270, labelpad=15)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_g_force_analysis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        # Option 1: Run simulation without obstacles (default)
        print("\n" + "="*60)
        print("F15 SIMULATION - WITHOUT OBSTACLES")
        print("="*60)
        flight_data = run_simulation()
        create_plots(flight_data)
        create_command_plots(flight_data)
        create_control_surface_position_plots(flight_data)
        create_g_force_plots(flight_data)
        
        # Option 2: Run simulation with obstacles
        print("\n" + "="*60)
        print("F15 SIMULATION - WITH OBSTACLE AVOIDANCE")
        print("="*60)
        print(f"Obstacles defined:")
        for i, obs in enumerate(EXAMPLE_OBSTACLES):
            print(f"  Obstacle {i}: Lat {obs['lat']:.3f}, Lon {obs['lon']:.3f}, Radius {obs['radius_m']}m")
        print()
        
        flight_data_obstacles = run_simulation(EXAMPLE_OBSTACLES)
        create_plots(flight_data_obstacles, EXAMPLE_OBSTACLES)
        create_command_plots(flight_data_obstacles)
        create_control_surface_position_plots(flight_data_obstacles)
        create_g_force_plots(flight_data_obstacles)
        
        print("\nPlots saved:")
        print("  - Flight parameters: 'f15_takeoff_flight_simulation.png'")
        print("  - Command values: 'f15_command_values.png'")
        print("  - Control surface positions: 'f15_control_surface_positions.png'")
        print("  - G force analysis: 'f15_g_force_analysis.png'")
        print("\nObstacle Avoidance Features:")
        print("  - Cylindrical obstacles with infinite height")
        print("  - Altitude avoidance: Climbs above safe altitude when obstacles detected")
        print("  - Horizontal avoidance: Turns away from obstacles in flight path")
        print("  - Real-time threat assessment based on aircraft heading and position")
        print("  - Configurable safety margins and avoidance distances")
        
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()
