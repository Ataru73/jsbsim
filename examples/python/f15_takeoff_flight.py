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

def run_simulation():
    """Run the F15 takeoff and flight simulation"""
    fdm = initialize_f15()
    
    data = {
        'time': [], 'altitude': [], 'airspeed': [], 'roll': [], 'aileron': []
    }
    
    simulation_time = 300.0
    
    # --- State Tracking Flags (Restored to original working logic) ---
    engines_started = False
    takeoff_roll_started = False
    airborne = False
    gear_retracted = False
    
    # --- Controller State Variables ---
    altitude_pid_integral = 0.0
    filtered_roll_rate_dps = 0.0
    roll_integral = 0.0

    print(f"\nStarting F15 simulation...")
    
    while fdm.get_sim_time() < simulation_time:
        sim_time = fdm.get_sim_time()

        # --- Takeoff Sequence (Restored to original time-based logic) ---
        if sim_time >= 0.1:
            fdm['fcs/flap-cmd-norm'] = 0.3 # Deploy flaps
            
        if sim_time >= 0.2:
            fdm['fcs/left-brake-cmd-norm'] = 0.0 # Release brakes
            fdm['fcs/right-brake-cmd-norm'] = 0.0
            
        if sim_time >= 0.3:
            fdm['propulsion/starter_cmd'] = 1 # Engage starter
            
        if sim_time >= 0.4 and not engines_started:
            fdm['propulsion/cutoff_cmd'] = 0 # Introduce fuel
            engines_started = True
            print(f"[{sim_time:5.1f}s] Engines started.")
            
        if sim_time >= 1.0 and not takeoff_roll_started:
            fdm['fcs/throttle-cmd-norm[0]'] = 1.0 # Full throttle
            fdm['fcs/throttle-cmd-norm[1]'] = 1.0
            takeoff_roll_started = True
            print(f"[{sim_time:5.1f}s] Full throttle - takeoff roll.")
            
        if takeoff_roll_started and not airborne and fdm['velocities/vc-kts'] > 120:
            fdm['fcs/elevator-cmd-norm'] = -0.4 # Rotate
            
        if takeoff_roll_started and not airborne and fdm['position/h-agl-ft'] > 20:
            airborne = True
            print(f"[{sim_time:5.1f}s] Airborne.")
            
        if airborne and fdm.get_sim_time() > 45.0 and not gear_retracted:
            fdm['fcs/flap-cmd-norm'] = 0.0 # Retract flaps
            fdm['gear/gear-cmd-norm'] = 0 # Retract gear
            gear_retracted = True
            print(f"[{sim_time:5.1f}s] Gear and flaps retracted.")

        # --- In-Flight Control Logic ---
        if airborne:
            if sim_time >= 70.0:
                target_altitude, target_airspeed = 5000, 350
            else:
                target_altitude, target_airspeed = 5000, 300
            
            # Altitude Controller
            altitude_error = target_altitude - fdm['position/h-agl-ft']
            altitude_pid_integral += altitude_error * DT
            altitude_pid_integral = max(-2000, min(2000, altitude_pid_integral)) # Anti-windup
            
            elevator_cmd = -0.0007 * altitude_error + 0.00001 * altitude_pid_integral - 0.015 * (-fdm['velocities/h-dot-fps'])
            fdm['fcs/elevator-cmd-norm'] = max(-0.6, min(0.3, elevator_cmd))
            
            # Speed Controller
            airspeed_error = target_airspeed - fdm['velocities/vc-kts']
            if altitude_error > 500 or airspeed_error > 25:
                throttle_cmd = 1.0
            else:
                throttle_cmd = 0.75
            fdm['fcs/throttle-cmd-norm[0]'] = throttle_cmd
            fdm['fcs/throttle-cmd-norm[1]'] = throttle_cmd
            
            # --- Aileron Controller (Filtered PID for Roll Stability) ---
            roll_angle_deg = fdm['attitude/phi-deg']
            raw_roll_rate_dps = fdm['velocities/p-rad_sec'] * 57.2958

            alpha = 0.2
            filtered_roll_rate_dps = (alpha * raw_roll_rate_dps) + ((1 - alpha) * filtered_roll_rate_dps)

            roll_integral += roll_angle_deg * DT
            roll_integral = max(-15.0, min(15.0, roll_integral))

            kp_roll, ki_roll, kd_roll = 0.02, 0.004, 0.15

            aileron_cmd = -(
                (kp_roll * roll_angle_deg) +
                (ki_roll * roll_integral) +
                (kd_roll * filtered_roll_rate_dps)
            )
            
            fdm['fcs/aileron-cmd-norm'] = max(-0.25, min(0.25, aileron_cmd))

        # --- Run Simulation Step and Collect Data ---
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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