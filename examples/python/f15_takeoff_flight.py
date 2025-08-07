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
DT = 0.0083333

# Define a list of waypoints to follow
# Each waypoint is a dictionary: {'lat': latitude, 'lon': longitude, 'alt': altitude_ft, 'speed': speed_kts}
WAYPOINTS = [
    {'lat': 41.70, 'lon': 12.45, 'alt': 5000, 'speed': 300},
    {'lat': 41.75, 'lon': 12.55, 'alt': 6000, 'speed': 320},
    {'lat': 41.70, 'lon': 12.65, 'alt': 5000, 'speed': 300},
    {'lat': 41.65, 'lon': 12.55, 'alt': 4000, 'speed': 280},
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
    def __init__(self, fdm, waypoints):
        self.fdm = fdm
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.engines_started = False
        self.takeoff_roll_started = False
        self.airborne = False
        self.gear_retracted = False
        self.altitude_pid_integral = 0.0
        self.filtered_roll_rate_dps = 0.0
        self.roll_integral = 0.0
        
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

    def execute_in_flight_control(self, sim_time):
        if self.airborne:
            # Waypoint Navigation Logic
            if self.current_waypoint_idx < len(self.waypoints):
                current_wp = self.waypoints[self.current_waypoint_idx]
                lat, lon = self.fdm['position/lat-gc-deg'], self.fdm['position/long-gc-deg']
                dist_to_wp = self.calculate_distance_m(lat, lon, current_wp['lat'], current_wp['lon'])

                if dist_to_wp < 1000:  # Waypoint capture distance in meters
                    print(f"[{sim_time:5.1f}s] Reached Waypoint {self.current_waypoint_idx}. Distance: {dist_to_wp:.2f}m")
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx >= len(self.waypoints):
                        print(f"[{sim_time:5.1f}s] Final waypoint reached. Maintaining course.")
                    else:
                        next_wp = self.waypoints[self.current_waypoint_idx]
                        print(f"[{sim_time:5.1f}s] Heading to Waypoint {self.current_waypoint_idx} at {next_wp['alt']} ft, {next_wp['speed']} kts.")

            # Determine target parameters
            if self.current_waypoint_idx < len(self.waypoints):
                # Navigate to the current waypoint
                target_wp = self.waypoints[self.current_waypoint_idx]
                lat, lon = self.fdm['position/lat-gc-deg'], self.fdm['position/long-gc-deg']
                target_bearing = self.calculate_bearing(lat, lon, target_wp['lat'], target_wp['lon'])
                heading = self.fdm['attitude/psi-deg']
                heading_error = (target_bearing - heading + 180) % 360 - 180
                target_roll_deg = max(-45, min(45, heading_error * 1.5))
                target_altitude = target_wp['alt']
                target_airspeed = target_wp['speed']
            else:
                # Last waypoint reached, maintain current heading and last altitude/speed
                target_roll_deg = 0  # Level wings
                last_wp = self.waypoints[-1]
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
            self.altitude_pid_integral += altitude_error * DT
            self.altitude_pid_integral = max(-2000, min(2000, self.altitude_pid_integral))
            elevator_cmd = -0.0007 * altitude_error + 0.00001 * self.altitude_pid_integral - 0.015 * (-self.fdm['velocities/h-dot-fps'])
            self.fdm['fcs/elevator-cmd-norm'] = max(-0.6, min(0.3, elevator_cmd))
            
            airspeed_error = target_airspeed - self.fdm['velocities/vc-kts']
            throttle_cmd = 1.0 if altitude_error > 500 or airspeed_error > 25 else 0.75
            self.fdm['fcs/throttle-cmd-norm[0]'] = throttle_cmd
            self.fdm['fcs/throttle-cmd-norm[1]'] = throttle_cmd

def run_simulation():
    """Run the F15 takeoff and flight simulation"""
    fdm = initialize_f15()
    autopilot = F15AutoPilot(fdm, WAYPOINTS)

    data = {
        'time': [], 'altitude': [], 'airspeed': [], 'roll': [], 'aileron': [],
        'lat': [], 'lon': []
    }
    simulation_time = 600.0
    
    print(f"\nStarting F15 simulation...")
    
    while fdm.get_sim_time() < simulation_time:
        sim_time = fdm.get_sim_time()
        autopilot.execute_takeoff_sequence(sim_time)
        autopilot.execute_in_flight_control(sim_time)

        fdm.run()
        
        data['time'].append(sim_time)
        data['altitude'].append(fdm['position/h-agl-ft'])
        data['airspeed'].append(fdm['velocities/vc-kts'])
        data['roll'].append(fdm['attitude/phi-deg'])
        data['aileron'].append(fdm['fcs/aileron-cmd-norm'])
        data['lat'].append(fdm['position/lat-gc-deg'])
        data['lon'].append(fdm['position/long-gc-deg'])
    
    print("\nSimulation completed!")
    return data

def create_plots(data):
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
    ax4.set_title('Ground Track')
    ax4.set_ylabel('Latitude (deg)')
    ax4.set_xlabel('Longitude (deg)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_takeoff_flight_simulation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        flight_data = run_simulation()
        create_plots(flight_data)
        print("\nPlot saved as 'f15_takeoff_flight_simulation.png'")
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()