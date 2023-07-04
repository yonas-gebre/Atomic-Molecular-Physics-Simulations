import sys
import numpy as np
from math import pi, sqrt, log, ceil
idx =  [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]

sys.path.append(path_address + '/General_Functions')
import Module as Mod

def Chirped_Gaussian(time, tau, beta, center = 0):
    argument = -2*np.log(2)/(1+beta*beta)*np.power((time - center)/tau, 2.0)
    return np.exp(argument)

def Chirped_Omega_Phi(omega, time, tau, beta, center = 0):
    omega = omega + 4*np.log(2)*beta/(1+beta*beta)*(time - center)/(tau*tau)
    phi = omega*(time - center) + 2*np.log(2)*beta/(1+beta*beta)*np.power((time - center)/tau,2)
    return omega, phi

def Gaussian(time, tau, center = 0):
    argument = -1*np.log(2)*np.power(2 *(time - center)/tau, 2.0)
    return np.exp(argument)

def Sin(time, tau, center = 0):
    return np.power(np.sin(pi*time / tau), 2.0)

def Asymmetric_Sin(time, tau, n, m, center = 0):
    return np.append(np.power(np.sin(pi*time[0:int(len(time)/2)] / tau), n),  np.power(np.sin(pi*time[int(len(time)/2):] / tau), m) )

def Freq_Shift(omega, envelop_fun, num_of_cycles, freq_shift = 1):
    if freq_shift == 1:
        if envelop_fun == Sin or envelop_fun == Asymmetric_Sin:
            mu = 4*pow(np.arcsin(np.exp(-1/4)),2.0)
        if envelop_fun == Gaussian or Chirped_Gaussian:
            mu = 8*np.log(2.0)/pow(pi,2)
        omega = omega*2.0/(1.0 + np.sqrt(1+ mu/pow(num_of_cycles,2))) 
        return omega
    elif freq_shift == 0:
        return omega
    else:
        print(freq_shift)
        print("freq shift should be 1 for true and 0 for false")
        exit()

def Cycle_to_time(cycle, omega):
    return 2*pi*cycle/omega

def Laser_Vectors(polarization, poynting):
    polarization = polarization / np.linalg.norm(polarization)
    poynting = poynting / np.linalg.norm(poynting)
    
    if abs(np.dot(polarization, poynting)) >= 1e-15:
        print("polarization of laser and the poynting direction are not orthagonal")
        exit()

    ellipticity_Vector = np.cross(polarization, poynting) 
    ellipticity_Vector = ellipticity_Vector / np.linalg.norm(ellipticity_Vector)

    return polarization, poynting, ellipticity_Vector

def Pulse(intensity, envelop_fun, omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, beta, freq_shift = 1, cycle_delay = 0, gaussian_length = 5, sin_ramp_up = 2, sin_ramp_down = 2):
    omega = Freq_Shift(omega, envelop_fun, num_of_cycles, freq_shift)
    tau = Cycle_to_time(num_of_cycles, omega)
    tau_delay = Cycle_to_time(cycle_delay, omega)

    
    amplitude = pow(intensity/3.51e16, 0.5) / omega
    chirped_gaussian_amplitude = amplitude*pow((1+beta*beta), -1/4)
    chirped_gaussian_time = np.arange(0,  gaussian_length* tau , time_spacing)
    chirped_gaussian_omega, chirped_gaussian_phi = Chirped_Omega_Phi(omega, chirped_gaussian_time, tau, beta, (gaussian_length * tau)/2)
    polarization, poynting, ellipticity_Vector = Laser_Vectors(polarization, poynting)

    Electric_Field = {}
    Vector_Potential = {}

    for i in range(3):
    

        if envelop_fun == Chirped_Gaussian:
            time = chirped_gaussian_time
            envelop = chirped_gaussian_amplitude * envelop_fun(time, tau, beta, (gaussian_length * tau)/2)
            Vector_Potential[i] = envelop * 1/np.sqrt(1+pow(ellipticity,2.0)) * (polarization[i] * np.cos(chirped_gaussian_phi + CEP) - ellipticity * ellipticity_Vector[i] * np.sin(chirped_gaussian_phi + CEP))
            Electric_Field[i] =  -1.0 * np.gradient(Vector_Potential[i], time_spacing)

        else:
            if envelop_fun == Gaussian:
                time = np.arange(0, gaussian_length* tau, time_spacing)
                envelop = amplitude * envelop_fun(time, tau, (gaussian_length * tau)/2)
                
            if envelop_fun == Sin:  
                time = np.arange(0, tau + time_spacing, time_spacing)
                envelop = amplitude * envelop_fun(time, tau)

            if  envelop_fun == Asymmetric_Sin:  
                time = np.arange(0, tau + time_spacing, time_spacing)
                envelop = amplitude * envelop_fun(time, tau, sin_ramp_up, sin_ramp_down)

            Vector_Potential[i] = envelop * 1/np.sqrt(1+pow(ellipticity,2.0)) * (polarization[i] * np.sin(omega*(time - tau/2) + CEP) - ellipticity * ellipticity_Vector[i] * np.cos(omega*(time - tau/2) + CEP))
            Electric_Field[i] = -1.0 * np.gradient(Vector_Potential[i], time_spacing)
            
            # # #remove this
            # center = (gaussian_length * tau)/2
            # time = np.arange(0, gaussian_length* tau, time_spacing)
            # Electric_Field[i] =  amplitude*np.exp(-2*np.log(2)*np.power((time - center)/tau, 2.0)) * 1/np.sqrt(1+pow(ellipticity,2.0)) * (polarization[i] * np.cos(omega*time + CEP) - ellipticity * ellipticity_Vector[i] * np.sin(omega*time + CEP))
            # Vector_Potential[i] = Electric_Field[i]
    
            
        if tau_delay == 0 or tau_delay != 0:
            if envelop_fun == Gaussian:
                time_delay = np.arange(0, gaussian_length* tau_delay + time_spacing, time_spacing)
                Vector_Potential[i] = np.pad(Vector_Potential[i], (len(time_delay),0), 'constant', constant_values=(0,0))
                Electric_Field[i] = np.pad(Electric_Field[i], (len(time_delay),0), 'constant', constant_values=(0,0))
                time = np.linspace(0, gaussian_length * (tau+ tau_delay), len(Vector_Potential[i]))

            if envelop_fun == Sin or envelop_fun == Asymmetric_Sin:  
                time_delay = np.arange(0, tau_delay + time_spacing, time_spacing)
                Vector_Potential[i] = np.pad(Vector_Potential[i], (len(time_delay),0), 'constant', constant_values=(0,0))
                Electric_Field[i] = np.pad(Electric_Field[i], (len(time_delay),0), 'constant', constant_values=(0,0))
                time = np.linspace(0, tau + tau_delay, len(Vector_Potential[i]))
                
                
    return time, Electric_Field, Vector_Potential

def Build_Laser_Pulse(input_par):

    number_of_lasers = len(input_par["laser"]["pulses"])
    time_spacing = input_par["time_spacing"]

    laser = []
    total_polarization = np.zeros(3)
    total_poynting = np.zeros(3)
    laser_time = np.zeros(1)

    elliptical_pulse = False
    freq_shift = input_par["laser"]["freq_shift"]
    free_prop_steps = input_par["laser"]["free_prop_steps"]
    gaussian_length = input_par["laser"]["gaussian_length"]
    sin_ramp_up = input_par["laser"]["sin_ramp_up"]
    sin_ramp_down = input_par["laser"]["sin_ramp_down"] 

    for i in range(number_of_lasers):
        current_pulse = input_par["laser"]["pulses"][i]
        intensity = current_pulse["intensity"]
        envelop_fun = current_pulse["envelop"]
        omega = current_pulse["omega"]
        CEP = current_pulse["CEP"]
        num_of_cycles = current_pulse["num_of_cycles"]
        polarization = np.array(current_pulse["polarization"])
        poynting = np.array(current_pulse["poynting"])
        ellipticity = current_pulse["ellipticity"]
        cycle_delay = current_pulse["cycle_delay"]
        beta = current_pulse["beta"]
        

        if ellipticity != 0:
            elliptical_pulse = True
        time, Electric_Field, Vector_Potential = Pulse(intensity, eval(envelop_fun), omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, beta,  freq_shift, cycle_delay, gaussian_length, sin_ramp_up, sin_ramp_down)
    
        if len(time) > len(laser_time):
            laser_time = time

        total_polarization += polarization
        total_poynting += poynting
        laser.append({})
        
        laser[i] = {}


        if(input_par["gauge"] == "Length"):
            laser[i]['x'] = Electric_Field[0]
            laser[i]['y'] = Electric_Field[1]
            laser[i]['z'] = Electric_Field[2]
        elif(input_par["gauge"] == "Velocity"):
            laser[i]['x'] = Vector_Potential[0]
            laser[i]['y'] = Vector_Potential[1]
            laser[i]['z'] = Vector_Potential[2]
        else:
            print("Gauge not specified")  

    laser_pulse = {}
    laser_pulse['x'] = np.zeros(len(laser_time))
    laser_pulse['y'] = np.zeros(len(laser_time))
    laser_pulse['z'] = np.zeros(len(laser_time))

    for i in range(number_of_lasers):
        laser_pulse['x'][:len(laser[i]['x'])] +=  laser[i]['x']
        laser_pulse['y'][:len(laser[i]['y'])] +=  laser[i]['y']
        laser_pulse['z'][:len(laser[i]['z'])] +=  laser[i]['z']
    
    
    laser_pulse['Right'] = (laser_pulse['x'] + 1.0j*laser_pulse['y'])/2
    laser_pulse['Left'] =  (laser_pulse['x'] - 1.0j*laser_pulse['y'])/2


    total_polarization /= np.linalg.norm(total_polarization)
    total_poynting /= np.linalg.norm(total_poynting)
    
    free_prop_idx = len(laser_time) - 1
 
    if free_prop_steps != 0:
        laser_pulse['x'] = np.pad(laser_pulse['x'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['y'] = np.pad(laser_pulse['y'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['z'] = np.pad(laser_pulse['z'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['Right'] = np.pad(laser_pulse['Right'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['Left'] = np.pad(laser_pulse['Left'], (0, free_prop_steps), 'constant', constant_values=(0,0))

    laser_end_time = laser_time[-1] + free_prop_steps*time_spacing
    laser_time = np.linspace(0, laser_end_time, len(laser_pulse['x']))
    
    # Pulse_Plotter(laser_time, laser_pulse, "Pulse.png")

    
    return laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx

def Pulse_Plotter(laser_time, laser_pulse, plot_name):

    plt.plot(laser_time, laser_pulse['x'], label = 'x')
    plt.plot(laser_time, laser_pulse['y'], label = 'y')
    plt.plot(laser_time, laser_pulse['z'], label = 'z')
    
    plt.legend()
    plt.savefig(plot_name)
    plt.clf()

if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx = Build_Laser_Pulse(input_par)    
    
    print("Here is the length of the pulse in time steps: ", free_prop_idx)
    
    # polarization = np.array([1,0,0])
    # poynting = np.array([0,0,1])
    # time, Electric_Field, Vector_Potential = Pulse(1.0e14, Sin, 0.114, 2, np.pi/2, 0.1, polarization, poynting, 1, 0)
    # plt.plot(time, Vector_Potential[0])
    # # plt.savefig("pulse.png")
    # # plt.clf()
    # plt.show()

    
    

