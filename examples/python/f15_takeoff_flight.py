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

# Configuration
AIRCRAFT_NAME = "f15"
PATH_TO_JSBSIM_FILES = "../.."
DT = 0.0083333

def initialize_f15():
    """Initialize JSBSim with F15"""
    jsbsim.FGJSBBase().debug_lvl = 0
    fdm = jsbsim.FGFDMExec(PATH_TO_JSBSIM_FILES)
    fdm.load_model(AIRCRAFT_NAME)
    fdm.set_dt(DT)
    
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
    def __init__(self, fdm):
        self.fdm = fdm
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

    def execute_in_flight_control(self, sim_time):
        if self.airborne:
            target_altitude, target_airspeed = (5000, 350) if sim_time >= 70.0 else (5000, 300)
            altitude_error = target_altitude - self.fdm['position/h-agl-ft']
            self.altitude_pid_integral += altitude_error * DT
            self.altitude_pid_integral = max(-2000, min(2000, self.altitude_pid_integral))
            elevator_cmd = -0.0007 * altitude_error + 0.00001 * self.altitude_pid_integral - 0.015 * (-self.fdm['velocities/h-dot-fps'])
            self.fdm['fcs/elevator-cmd-norm'] = max(-0.6, min(0.3, elevator_cmd))
            airspeed_error = target_airspeed - self.fdm['velocities/vc-kts']
            throttle_cmd = 1.0 if altitude_error > 500 or airspeed_error > 25 else 0.75
            self.fdm['fcs/throttle-cmd-norm[0]'] = throttle_cmd
            self.fdm['fcs/throttle-cmd-norm[1]'] = throttle_cmd
            roll_angle_deg = self.fdm['attitude/phi-deg']
            raw_roll_rate_dps = self.fdm['velocities/p-rad_sec'] * 57.2958
            alpha = 0.2
            self.filtered_roll_rate_dps = (alpha * raw_roll_rate_dps) + ((1 - alpha) * self.filtered_roll_rate_dps)
            self.roll_integral += roll_angle_deg * DT
            self.roll_integral = max(-15.0, min(15.0, self.roll_integral))
            kp_roll, ki_roll, kd_roll = 0.02, 0.004, 0.15
            aileron_cmd = -(
                (kp_roll * roll_angle_deg) +
                (ki_roll * self.roll_integral) +
                (kd_roll * self.filtered_roll_rate_dps)
            )
            self.fdm['fcs/aileron-cmd-norm'] = max(-0.25, min(0.25, aileron_cmd))

def run_simulation():
    """Run the F15 takeoff and flight simulation"""
    fdm = initialize_f15()
    autopilot = F15AutoPilot(fdm)

    data = {
        'time': [], 'altitude': [], 'airspeed': [], 'roll': [], 'aileron': []
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
    
    print("\nSimulation completed!")
    return data

def create_plots(data):
    """Create plots of the key flight parameters."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('F15 Eagle - Stable Flight Simulation', fontsize=16, fontweight='bold')
    times = data['time']

    # Altitude
    ax1.plot(times, data['altitude'], 'b-', linewidth=2)
    ax1.set_title('Altitude Profile')
    ax1.set_ylabel('Altitude (ft AGL)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Airspeed
    ax2.plot(times, data['airspeed'], 'r-', linewidth=2)
    ax2.set_title('Airspeed Profile')
    ax2.set_ylabel('Speed (kts)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Roll Angle
    ax3.plot(times, data['roll'], 'r-', linewidth=2)
    ax3.set_title('Roll Attitude')
    ax3.set_ylabel('Roll Angle (degrees)')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Aileron Command
    ax4.plot(times, data['aileron'], 'purple', linewidth=2)
    ax4.set_title('Aileron Control Command (Filtered PID)')
    ax4.set_ylabel('Aileron Position (-1 to 1)')
    ax4.set_xlabel('Time (seconds)')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('f15_takeoff_flight_simulation_STABLE.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        flight_data = run_simulation()
        create_plots(flight_data)
        print("\nPlot saved as 'f15_takeoff_flight_simulation_STABLE.png'")
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()