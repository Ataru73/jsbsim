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
    # Suppress debug messages
    jsbsim.FGJSBBase().debug_lvl = 0
    
    # Create FDM
    fdm = jsbsim.FGFDMExec(PATH_TO_JSBSIM_FILES)
    fdm.load_model(AIRCRAFT_NAME)
    fdm.set_dt(DT)
    
    # Set location (Pratica di Mare, Italy - as in Pilot.py)
    fdm['ic/lat-gc-deg'] = 41.658
    fdm['ic/long-gc-deg'] = 12.446
    fdm['ic/terrain-elevation-ft'] = 160
    
    # F15 initial conditions - on runway ready for takeoff
    fdm['ic/h-agl-ft'] = 5  # Slightly above ground to prevent ground contact issues
    fdm['ic/altitude-gnd-ft'] = 10
    fdm['ic/psi-true-deg'] = 180.0  # Heading south
    fdm['ic/vc-kts'] = 0.0  # Stationary
    fdm['ic/phi-deg'] = 0.0  # Wings level
    fdm['ic/theta-deg'] = 0.0  # Level attitude
    fdm['ic/beta-deg'] = 0.0  # No sideslip
    
    # Initialize
    fdm.run_ic()
    
    print(f"Aircraft initialized: {AIRCRAFT_NAME}")
    print(f"Location: {fdm['ic/lat-gc-deg']:.3f}N, {fdm['ic/long-gc-deg']:.3f}E")
    print(f"Initial altitude: {fdm['position/h-agl-ft']:.1f} ft AGL")
    
    return fdm

def run_simulation():
    """Run the F15 takeoff and flight simulation"""
    fdm = initialize_f15()
    
    # Data storage
    data = {
        'time': [],
        'altitude': [],
        'airspeed': [],
        'groundspeed': [],
        'heading': [],
        'pitch': [],
        'roll': [],
        'throttle_1': [],
        'throttle_2': [],
        'elevator': [],
        'aileron': [],
        'flaps': [],
        'gear': [],
        'phase': []
    }
    
    # Flight phases
    PHASE_STARTUP = "Engine Start"
    PHASE_TAXI = "Taxi"
    PHASE_TAKEOFF_ROLL = "Takeoff Roll"
    PHASE_ROTATION = "Rotation" 
    PHASE_INITIAL_CLIMB = "Initial Climb"
    PHASE_CRUISE_CLIMB = "Cruise Climb"
    PHASE_LEVEL_FLIGHT = "Level Flight"
    
    current_phase = PHASE_STARTUP
    
    # Simulation parameters
    simulation_time = 300.0  # 5 minutes
    step_count = 0
    
    # Flight state tracking
    engines_started = False
    takeoff_roll_started = False
    airborne = False
    gear_retracted = False
    flaps_retracted = False
    
    # --- Controller State Variables ---
    altitude_pid_integral = 0.0
    altitude_pid_prev_error = 0.0
    filtered_roll_rate_dps = 0.0 # State for the roll rate filter
    filtered_roll_angle_deg = 0.0 # State for the roll angle filter
    
    print(f"\nStarting F15 simulation...")
    print("=" * 50)
    
    while fdm.get_sim_time() < simulation_time:
        sim_time = fdm.get_sim_time()
        current_altitude = fdm['position/h-agl-ft']
        current_airspeed = fdm['velocities/vc-kts']
        
        # ============= FLIGHT PHASE LOGIC =============
        
        # Engine startup sequence (following Pilot.py pattern)
        if sim_time >= 0.1 and not engines_started:
            # Deploy flaps for takeoff
            fdm['fcs/flap-cmd-norm'] = 0.3
            print(f"[{sim_time:5.1f}s] Deploying flaps for takeoff")
            
        if sim_time >= 0.2:
            # Release brakes
            fdm['fcs/left-brake-cmd-norm'] = 0.0
            fdm['fcs/right-brake-cmd-norm'] = 0.0
            
        if sim_time >= 0.3:
            # Start engines
            fdm['propulsion/starter_cmd'] = 1
            
        if sim_time >= 0.4 and not engines_started:
            # Ensure engines are running
            fdm['propulsion/cutoff_cmd'] = 0
            engines_started = True
            current_phase = PHASE_TAXI
            print(f"[{sim_time:5.1f}s] Engines started")
            
        # Takeoff roll
        if sim_time >= 1.0 and not takeoff_roll_started:
            # Full throttle for takeoff (both engines)
            fdm['fcs/throttle-cmd-norm[0]'] = 1.0
            fdm['fcs/throttle-cmd-norm[1]'] = 1.0
            takeoff_roll_started = True
            current_phase = PHASE_TAKEOFF_ROLL
            print(f"[{sim_time:5.1f}s] Full throttle - beginning takeoff roll")
            
        # Rotation - based on airspeed rather than time
        if takeoff_roll_started and not airborne and current_airspeed > 120:  # F15 rotation speed
            # Pitch up for takeoff
            fdm['fcs/elevator-cmd-norm'] = -0.4  # Strong back pressure for rotation
            current_phase = PHASE_ROTATION
            print(f"[{sim_time:5.1f}s] Rotation at {current_airspeed:.1f} kts")
            
        # Check if airborne (separate from rotation to ensure it gets checked every frame)
        if takeoff_roll_started and not airborne and current_altitude > 20:  # 20 feet AGL - airborne
            airborne = True
            current_phase = PHASE_INITIAL_CLIMB
            print(f"[{sim_time:5.1f}s] Airborne at {current_airspeed:.1f} kts")
                
        # Gear and flap retraction after takeoff
        if airborne and sim_time >= 45.0 and not flaps_retracted:
            # Retract flaps
            fdm['fcs/flap-cmd-norm'] = 0.0
            flaps_retracted = True
            print(f"[{sim_time:5.1f}s] Retracting flaps")
            
        if airborne and sim_time >= 46.0 and not gear_retracted:
            # Retract landing gear
            fdm['gear/gear-cmd-norm'] = 0
            gear_retracted = True
            print(f"[{sim_time:5.1f}s] Retracting landing gear")
            
        # Initial throttle reduction after takeoff (will be overridden by flight control)
        if airborne and sim_time >= 47.0:
            current_phase = PHASE_CRUISE_CLIMB
            
        # Manual flight control throughout the simulation
        if airborne:
            # Get current flight parameters
            altitude_rate = fdm['velocities/h-dot-fps']
            
            # Determine flight phase and set target altitude
            if sim_time >= 70.0:
                current_phase = PHASE_LEVEL_FLIGHT
                target_altitude = 5000  # Higher cruising altitude
                target_airspeed = 350  # Fighter jet cruise speed
            elif sim_time >= 30.0:  # Earlier transition to cruise climb
                current_phase = PHASE_CRUISE_CLIMB
                target_altitude = 5000  # Climb to higher altitude
                target_airspeed = 300  # Fast climb speed
            else:
                target_altitude = 3000  # Initial climb target
                target_airspeed = 250
            
            # Altitude control using a PID controller
            altitude_error = target_altitude - current_altitude
            airspeed_error = target_airspeed - current_airspeed

            # PID gains (tune as needed)
            kp_alt = -0.0007  # Proportional gain
            ki_alt = 0.00001  # Integral gain
            kd_alt = -0.015  # Derivative gain

            # PID calculations
            altitude_pid_integral += altitude_error * DT
            altitude_pid_integral = max(-2000, min(2000, altitude_pid_integral)) # Anti-windup
            
            # PID output
            elevator_cmd = (
                kp_alt * altitude_error
                + ki_alt * altitude_pid_integral
                + kd_alt * (-altitude_rate)
            )
            altitude_pid_prev_error = altitude_error
            elevator_cmd = max(-0.6, min(0.3, elevator_cmd))
            fdm['fcs/elevator-cmd-norm'] = elevator_cmd
            
            # Throttle control for airspeed and climb
            if altitude_error > 1000:
                throttle_cmd = 1.0
            elif airspeed_error > 50:
                throttle_cmd = 0.9
            elif airspeed_error > -20 or altitude_error > 100:
                throttle_cmd = 0.8 if sim_time >= 70.0 else 0.7
            else:
                throttle_cmd = 0.6
                
            fdm['fcs/throttle-cmd-norm[0]'] = throttle_cmd
            fdm['fcs/throttle-cmd-norm[1]'] = throttle_cmd
            
            # Simple heading hold
            heading_error = 180.0 - fdm['attitude/psi-deg']
            if heading_error > 180: heading_error -= 360
            elif heading_error < -180: heading_error += 360
            rudder_cmd = max(-0.5, min(0.5, heading_error * 0.01))
            fdm['fcs/rudder-cmd-norm'] = rudder_cmd
            
            # ====================================================================
            # === FILTERED PD WING LEVELER (FINAL STABLE VERSION) - REV 3      ===
            # ====================================================================
            # High-frequency chatter indicated derivative term was amplifying noise.
            # This version adds a low-pass filter to the roll rate signal before
            # it's used by the controller, ensuring smooth and stable corrections.
            
            # Get current roll angle and raw roll rate
            roll_angle_deg = fdm['attitude/phi-deg']
            raw_roll_rate_dps = fdm['velocities/p-rad_sec'] * 57.2958

            # Apply a low-pass filter to smooth the roll rate signal
            # A small alpha value results in more smoothing.
            alpha = 0.8
            filtered_roll_rate_dps = (alpha * raw_roll_rate_dps) + ((1 - alpha) * filtered_roll_rate_dps)
            filtered_roll_angle_deg = (alpha * roll_angle_deg) + ((1 - alpha) * filtered_roll_angle_deg)

            # PD Controller gains, re-tuned for the smooth, filtered signal
            kp_roll = 0.015  # Proportional gain
            kd_roll = 0.12   # Derivative gain (acts on the filtered rate)

            # Calculate the aileron command using the filtered rate
            aileron_cmd = -(kp_roll * filtered_roll_angle_deg + kd_roll * filtered_roll_rate_dps)
            
            # Limit the command to a conservative range
            aileron_cmd = max(-0.2, min(0.2, aileron_cmd))
            
            fdm['fcs/aileron-cmd-norm'] = aileron_cmd
            # ====================================================================
            # === END OF REVISED LOGIC                                         ===
            # ====================================================================

        # Run simulation step
        fdm.run()
        
        # Collect data
        if step_count % 5 == 0:
            data['time'].append(sim_time)
            data['altitude'].append(current_altitude)
            data['airspeed'].append(current_airspeed)
            data['groundspeed'].append(fdm['velocities/vg-fps'] * 0.592484)
            data['heading'].append(fdm['attitude/psi-deg'])
            data['pitch'].append(fdm['attitude/theta-deg'])
            data['roll'].append(fdm['attitude/phi-deg']) 
            data['throttle_1'].append(fdm['fcs/throttle-cmd-norm[0]'])
            data['throttle_2'].append(fdm['fcs/throttle-cmd-norm[1]'])
            data['elevator'].append(fdm['fcs/elevator-cmd-norm'])
            data['aileron'].append(fdm['fcs/aileron-cmd-norm'])
            data['flaps'].append(fdm['fcs/flap-cmd-norm'])
            data['gear'].append(fdm['gear/gear-cmd-norm'])
            data['phase'].append(current_phase)
        
        # Progress reporting
        if step_count % 600 == 0:
            print(f"[{sim_time:5.1f}s] {current_phase}: Alt={current_altitude:6.0f}ft, "
                  f"IAS={current_airspeed:5.1f}kts, HDG={fdm['attitude/psi-deg']:5.1f}°")
        
        step_count += 1
    
    print("\nSimulation completed!")
    return data

def create_plots(data):
    """Create comprehensive plots of the flight simulation"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('F15 Eagle - Takeoff and Steady Flight Simulation', fontsize=16, fontweight='bold')
    
    times = np.array(data['time'])
    
    # Plot 1: Altitude profile
    axes[0,0].plot(times, data['altitude'], 'b-', linewidth=2)
    axes[0,0].axhline(y=5000, color='r', linestyle='--', alpha=0.7, label='Target Alt')
    axes[0,0].set_ylabel('Altitude (ft AGL)')
    axes[0,0].set_title('Altitude Profile')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot 2: Airspeed
    axes[0,1].plot(times, data['airspeed'], 'r-', linewidth=2, label='IAS')
    axes[0,1].plot(times, data['groundspeed'], 'g-', linewidth=1, alpha=0.7, label='GS')
    axes[0,1].axhline(y=120, color='orange', linestyle='--', alpha=0.7, label='Rotation Speed')
    axes[0,1].set_ylabel('Speed (kts)')
    axes[0,1].set_title('Airspeed Profile')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot 3: Attitude angles
    axes[0,2].plot(times, data['pitch'], 'b-', linewidth=2, label='Pitch')
    axes[0,2].plot(times, data['roll'], 'r-', linewidth=1, label='Roll')
    axes[0,2].set_ylabel('Angle (degrees)')
    axes[0,2].set_title('Aircraft Attitude')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # Plot 4: Heading
    axes[1,0].plot(times, data['heading'], 'purple', linewidth=2)
    axes[1,0].set_ylabel('Heading (degrees)')
    axes[1,0].set_title('Aircraft Heading')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Engine throttles
    axes[1,1].plot(times, data['throttle_1'], 'g-', linewidth=2, label='Engine 1')
    axes[1,1].plot(times, data['throttle_2'], 'orange', linewidth=2, label='Engine 2')
    axes[1,1].set_ylabel('Throttle Position')
    axes[1,1].set_title('Engine Throttles')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # Plot 6: Flight controls
    axes[1,2].plot(times, data['elevator'], 'b-', linewidth=2, label='Elevator')
    axes[1,2].plot(times, data['aileron'], 'r-', linewidth=2, label='Aileron (Filtered PD)')
    axes[1,2].plot(times, data['flaps'], 'brown', linewidth=1, alpha=0.7, label='Flaps')
    axes[1,2].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Aileron Limit')
    axes[1,2].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    axes[1,2].set_ylabel('Control Position')
    axes[1,2].set_title('Flight Controls (Filtered PD Control)')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    
    # Plot 7: Gear position
    axes[2,0].plot(times, data['gear'], 'gray', linewidth=2)
    axes[2,0].set_ylabel('Gear Position (1=Down, 0=Up)')
    axes[2,0].set_xlabel('Time (seconds)')
    axes[2,0].set_title('Landing Gear')
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].set_ylim(-0.1, 1.1)
    
    # Plot 8: Flight phases timeline
    phase_map = {p: i for i, p in enumerate(sorted(list(set(data['phase']))))}
    phase_colors = [phase_map[p] for p in data['phase']]
    axes[2,1].scatter(times, [0]*len(times), c=phase_colors, cmap='viridis', s=20, marker='|')
    
    phase_changes = []
    if data['phase']:
        current_phase = data['phase'][0]
        phase_start = times[0]
        for i, phase in enumerate(data['phase']):
            if phase != current_phase:
                phase_changes.append({'start': phase_start, 'end': times[i-1], 'label': current_phase})
                current_phase = phase
                phase_start = times[i]
        phase_changes.append({'start': phase_start, 'end': times[-1], 'label': current_phase})

    for change in phase_changes:
        axes[2,1].text((change['start'] + change['end']) / 2, 0.05, change['label'], rotation=45, ha='center', va='bottom', fontsize=9)
    
    axes[2,1].set_xlabel('Time (seconds)')
    axes[2,1].set_yticks([])
    axes[2,1].set_title('Flight Phase Timeline')
    axes[2,1].set_ylim(-0.5, 0.5)

    # Plot 9: Summary stats
    axes[2,2].axis('off')
    stats_text = f"""**Flight Summary**
    Total time: {times[-1]:.1f}s
    Max altitude: {max(data['altitude']):.0f} ft
    Max airspeed: {max(data['airspeed']):.1f} kts
    Final altitude: {data['altitude'][-1]:.0f} ft
    Final airspeed: {data['airspeed'][-1]:.1f} kts
    Final heading: {data['heading'][-1]:.0f}°
    """
    axes[2,2].text(0.05, 0.95, stats_text, transform=axes[2,2].transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Set common x-axis labels
    for ax in [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]:
        ax.set_xlabel('')
    for ax in [axes[2,0], axes[2,1]]:
        ax.set_xlabel('Time (seconds)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('f15_takeoff_flight_simulation_STABLE.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        flight_data = run_simulation()
        if flight_data:
            create_plots(flight_data)
            print("\nF15 simulation completed successfully!")
            print("Plot saved as 'f15_takeoff_flight_simulation_STABLE.png'")
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()