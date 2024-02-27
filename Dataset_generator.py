########################################## DATASET GENERATOR ##################################################
# by Judith Beck



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

import astropy.units as u
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
import numpy as np
import os
import sys
from multiprocessing import Pool
import uuid
from tqdm import tqdm
import warnings
import subprocess
from datetime import datetime
import scipy.integrate as integrate

## get path of current directory
LocalPath = os.getcwd() + "/"  # Getting the current working directory and storing it in LocalPath

# get path of XCLASS directory
XCLASSRootDir = '/scratch/beck/XCLASS/XCLASS__version_1.4.1/'  # Setting the path to XCLASS directory

# extend sys.path variable
NewPath = XCLASSRootDir + "build_tasks/"  # Creating a new path to build_tasks directory within XCLASS
if (not NewPath in sys.path):  # Checking if the new path is not already in sys.path
    sys.path.append(NewPath)  # Appending the new path to sys.path

# import XCLASS packages
import task_myXCLASS  # Importing task_myXCLASS module from XCLASS packages

#original_stdout = sys.stdout
#sys.stdout = open('/dev/null', 'w')  # Redirecting standard output to /dev/null

if __name__ == "__main__":
    #------------------------------------------------------------------------------------------------------------
    ######################## VALUES DEFINED BY THE USER ########################################################
    #------------------------------------------------------------------------------------------------------------

    # Parameters of the artificial source

    single_choice = 'no'  # Setting whether single choice of parameters is selected
    grid_log = 'no'       # Setting whether grid space or logarithmic scaling is used
    random = 'yes'        # Setting whether random choice for every sample is selected

    ISO_RATIO = True      # Setting whether iso ratio is considered
    NOISE = False         # Setting whether noise variation is considered

    # Define the boundaries for each parameter
    n_min = 0.001  # Setting the minimum value for density parameter                          
    n_max = 100    # Setting the maximum value for density parameter

    temperature_min = 50      # Setting the minimum value for temperature parameter                                          
    temperature_max = 700     # Setting the maximum value for temperature parameter

    mass_protostar_min = 8    # Setting the minimum value for mass_protostar parameter                                     
    mass_protostar_max = 100  # Setting the maximum value for mass_protostar parameter

    power_temperature_min = 0.2     # Setting the minimum value for power_temperature parameter                                 
    power_temperature_max = 1.5     # Setting the maximum value for power_temperature parameter

    power_density_min = 0.5         # Setting the minimum value for power_density parameter                                     
    power_density_max = 2.5         # Setting the maximum value for power_density parameter

    iso_ratio_min = 20              # Setting the minimum value for iso_ratio parameter                                    
    iso_ratio_max = 100             # Setting the maximum value for iso_ratio parameter

    noise_min = 0.05                # Setting the minimum value for noise parameter                                       
    noise_max = 0.6                 # Setting the maximum value for noise parameter

    radius = 0.1  # Setting the radius of the sphere-shaped artificial source in pc 
    Radius = 0.001  # Setting the little radius at the center where temperature and density do not increase anymore

    radius_core_low = 0.005  # Setting the low value for radius_core parameter                                        
    radius_core_high = 0.008 # Setting the high value for radius_core parameter

    M_core_low = 0.1          # Setting the low value for M_core parameter                                               
    M_core_high = 100         # Setting the high value for M_core parameter

    L_M_low = 0.1             # Setting the low value for L_M parameter                                                  
    L_M_high = 5000           # Setting the high value for L_M parameter

    L_high = 10               # Setting the value for L_high parameter                                                    

    # Number of desired samples
    num_samples = 100  # Setting the number of desired samples
    
    if grid_log == 'yes':
        # Generate parameter values with grid space or logarithmic scaling
        Rho = np.logspace(np.log10(10**rho_log_min), np.log10(10**rho_log_max), n_samples) 
        Temperature = np.linspace(temperature_min, temperature_max, n_samples) 
        Mass_protostar = np.logspace(np.log10(10**mass_protostar_log_min), np.log10(10**mass_protostar_log_max), n_samples) 
        Power_temperature = np.linspace(power_temperature_min, power_temperature_max, n_samples) 
        Power_density = np.linspace(power_density_min, power_density_max, n_samples) 
        iso_ratio = np.linspace(iso_ratio_min, iso_ratio_max, n_samples)
        noise = np.linspace(noise_min, noise_max, n_samples)
        
        # Generate all combinations of the parameters
        combinations = list(itertools.product(Temperature, Power_temperature, Rho, Power_density, Mass_protostar, iso_ratio, noise, radius, Radius, radius_core))

    # Random choice for every sample  
    if random =='yes':
        # Set random seed for reproducibility
        np.random.seed(42)

        # Parsec conversions and other quantities
        pc_in_cm = 3.085677581491367*10**18 
        pc_in_m = 3.085677581491367*10**16 
        x_ch3cn = 1.e-8
        factor_theta_phi = 4. * np.pi

        M_Sun = 1.98847 * 10 ** 30 
        L_sun = 3.826e26
        x_ch3cn = 1.e-8
        sb= 5.670374419e-8

        # Function to calculate density integrand
        def get_density_integrand(r, n0, q):
            r0 = Radius * pc_in_cm
            density_value = n0 * (r/r0)**-q
            if r < r0:
                density_value = n0
            return density_value * r * r

        #Calculating the parameter-vectors 
        combinations = []

        i=0

        while i < num_samples:

            radius_core = np.random.uniform(radius_core_low, radius_core_high)
            lowerlim = 0
            upperlim = radius_core * pc_in_cm

            power_density = np.random.uniform(power_density_min, power_density_max)
            n = np.random.uniform(np.log10(n_min), np.log10(n_max))
            n = 10**n
            
            total_n_ch3cn = integrate.quad(get_density_integrand, lowerlim, upperlim, args=(n, power_density))[0] * factor_theta_phi
            total_n_h2 = total_n_ch3cn / x_ch3cn
            total_m_kg = total_n_h2 * 2.33 * 1.6735575e-27
            M_core = total_m_kg/M_Sun

            if M_core_low <= M_core <= M_core_high:

                attempt_count = 0  

                while attempt_count < 10:

                    temperature = np.random.uniform(temperature_min, temperature_max)
                    
                    mass_protostar = np.random.uniform(mass_protostar_min, mass_protostar_max)

                    L_core = (4*np.pi * sb * ((Radius * pc_in_m)**2) * (temperature**4))/L_sun
                    L_M_ratio = L_core/(M_core+mass_protostar)

                    if L_core >= L_high and L_M_low <= L_M_ratio <= L_M_high:

                        power_temperature = np.random.uniform(power_temperature_min, power_temperature_max)
                        iso_ratio = np.random.uniform(iso_ratio_min, iso_ratio_max)
                        noise = np.random.uniform(noise_min, noise_max)

                        if not ISO_RATIO and not NOISE:
                            combinations.append((temperature, power_temperature, n, power_density, mass_protostar, radius_core, radius, Radius))

                        if ISO_RATIO and not NOISE:
                            combinations.append((temperature, power_temperature, n, power_density, mass_protostar, iso_ratio, radius_core, radius, Radius))

                        if not ISO_RATIO and NOISE:
                            combinations.append((temperature, power_temperature, n, power_density, mass_protostar, noise, radius_core, radius, Radius))

                        if ISO_RATIO and NOISE:
                            combinations.append((temperature, power_temperature, n, power_density, mass_protostar, iso_ratio, noise, radius_core, radius, Radius))

                        i = i+1
                        break
                    else:
                        attempt_count += 1
                        continue

            else: 
                continue

        
    dataset = np.empty((0, 4302))  # Array for the finished dataset, empty at the beginning

    if not ISO_RATIO and not NOISE:
        parameterset = np.empty((0, 11)) # Array for the finished parameterset, empty at the beginning

    if ISO_RATIO and not NOISE:
        parameterset = np.empty((0, 12)) # Array for the finished parameterset, empty at the beginning

    if not ISO_RATIO and NOISE:
        parameterset = np.empty((0, 12)) # Array for the finished parameterset, empty at the beginning

    if ISO_RATIO and NOISE:
        parameterset = np.empty((0, 13)) # Array for the finished parameterset, empty at the beginning

    # Directory modifications
    directory_path = "/scratch/beck/XCLASS/temp/myXCLASS/"
    molfit_path = '/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/molfit_parallel/'

    # Command to delete remaining directory from last run and molfit file
    command1 = f"rm -r {directory_path}/*"
    command2 = f"rm -r {molfit_path}/*"

    # Execute commands in the terminal
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)


 #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################## LOOP OVER ALL COMBINATIONS OF THE PARAMETERS ##################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to process each combination of parameters
def process_combination(combination):

    # Set ISO_RATIO and NOISE flags
    ISO_RATIO = True
    NOISE = False

    # Unpack the combination of parameters based on ISO_RATIO and NOISE flags
    if not ISO_RATIO and not NOISE:
        temperature, power_temperature, rho, power_density, mass_protostar, radius_core, radius, Radius = combination

    if ISO_RATIO and not NOISE:
        temperature, power_temperature, rho, power_density, mass_protostar, iso_ratio, radius_core, radius, Radius = combination

    if not ISO_RATIO and NOISE:
        temperature, power_temperature, rho, power_density, mass_protostar, noise, radius_core, radius, Radius = combination

    if ISO_RATIO and NOISE:
        temperature, power_temperature, rho, power_density, mass_protostar, iso_ratio, noise, radius_core, radius, Radius = combination

    # Generate a unique ID for the current run
    run_id = str(uuid.uuid4())
    
#------------------------------------------------------------------------------------------------------------
######################## VALUES DEFINED BY THE USER ########################################################
#------------------------------------------------------------------------------------------------------------

    # Define filenames for Molfits and Isotopologue files
    MolfitsFileName = "/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/molfit_parallel/CH3CN_myXCLASS_%s.molfit" % run_id
    IsoFileName = "/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/molfit_parallel/CH3CN_iso_%s.txt" % run_id

    #-----------------------------------------------------------------------------------------------------------
    # Parameters for the spatial cube

    distance = 4400  # Distance of the source in parsec
    n_points = 121    # Number of pixels of the spatial cube in each direction (x, y, z)
    Restfrequency = 220.74726120 * u.GHz  # Set the rest frequency of CH3CN equal to the line centre (from splatalogue)
                                          # This sets the zero velocity to where the line would be with no relative motion

    #------------------------------------------------------------------------------------------------------------
    # Parameters for the spectral cube

    components = 31   # Number of components (in z-direction)
                      # Note: Components can also be set to n_points

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    
    ####################### QUANTITIES NEEDED IN THE CALCULATIONS ##################################################
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    G = 6.674 * 10 ** (-11)         # Gravitational constant
    M_Sun = 1.98847 * 10 ** 30      # Solar Mass
    pc_in_cm = 3.085677581491367*10**18  # Parsec in cm
    pc_in_m = 3.085677581491367*10**16    # Parsec in m
    x_ch3cn = 1.e-8
    sb= 5.670374419e-8               # Stefan-Boltzmann constant
    L_sun = 3.826e26                 # Luminosity of the Sun in watts

    # Pixel widths     
    pixel_width = 2*radius/n_points        # in pc
    pixel_width_cm = pixel_width * pc_in_cm    # in cm
    pixel_width_rad = np.arctan(pixel_width / distance)  # in rad
    pixel_width_deg = np.degrees(pixel_width_rad)        # in degree
    pixel_width_arcsec = pixel_width_deg * 3600           # in arcseconds

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    ############################# Iso ratio file #################################################################################
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Create contents for the Isotopologue file based on ISO_RATIO flag
    if ISO_RATIO == True:

        file_iso_contents = '''
        % isotopologue:                         Iso-master:                              Iso-ratio:              Lower-limit:              upper-limit:
        % CH3CN;v8=1;                           CH3CN;v=0;                               1                        1                         1
        C-13-H3CN;v=0;                          CH3CN;v=0;                               {}                       5                         100
        CH3C-13-N;v=0;                          CH3CN;v=0;                               {}                       5                         100
        '''

        file_iso_contents = file_iso_contents.format(iso_ratio, iso_ratio)

        # Write the Isotopologue file
        with open(IsoFileName, "w") as file:
            file.write(file_iso_contents)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    ############################# MyXCLASS paramters #################################################################################
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # define min. freq. (in MHz)
    FreqMin = 220193.7

    # define max. freq. (in MHz)
    FreqMax = 220767.3

    # define freq. step (in MHz)
    FreqStep = 0.4

    # depending on parameter "Inter_Flag" define beam size (in arcsec)
    # (Inter_Flag = True) or size of telescope (in m) (Inter_Flag = False)
    TelescopeSize = pixel_width_arcsec

    # define beam minor axis length (in arsec)
    BMIN = None

    # define beam major axis length (in arsec)
    BMAJ = None

    # define beam position angle (in degree)
    BPA = None

    # interferrometric data?
    Inter_Flag = True

    # define red shift
    Redshift = None

    # BACKGROUND: describe continuum with tBack and tslope only
    t_back_flag = True

    # BACKGROUND: define background temperature (in K)
    tBack = 0.0

    # BACKGROUND: define temperature slope (dimensionless)
    tslope = 0.0

    # BACKGROUND: define path and name of ASCII file describing continuum as function
    #             of frequency
    BackgroundFileName = ""

    # DUST: define hydrogen column density (in cm^(-2))
    N_H = 1.e24

    # DUST: define spectral index for dust (dimensionless)
    beta_dust = 0

    # DUST: define kappa at 1.3 mm (cm^(2) g^(-1))
    kappa_1300 = 0

    # DUST: define path and name of ASCII file describing dust opacity as
    #       function of frequency
    DustFileName = ""

    # FREE-FREE: define electronic temperature (in K)
    Te_ff = None

    # FREE-FREE: define emission measure (in pc cm^(-6))
    EM_ff = None

    # SYNCHROTRON: define kappa of energy spectrum of electrons (electrons m^(\u22123) GeV^(-1))
    kappa_sync = None

    # SYNCHROTRON: define magnetic field (in Gauss)
    B_sync = None

    # SYNCHROTRON: energy spectral index (dimensionless)
    p_sync = None

    # SYNCHROTRON: thickness of slab (in AU)
    l_sync = None

    # PHEN-CONT: define phenomenological function which is used to describe
    #            the continuum
    ContPhenFuncID = None

    # PHEN-CONT: define first parameter for phenomenological function
    ContPhenFuncParam1 = None

    # PHEN-CONT: define second parameter for phenomenological function
    ContPhenFuncParam2 = None

    # PHEN-CONT: define third parameter for phenomenological function
    ContPhenFuncParam3 = None

    # PHEN-CONT: define fourth parameter for phenomenological function
    ContPhenFuncParam4 = None

    # PHEN-CONT: define fifth parameter for phenomenological function
    ContPhenFuncParam5 = None

    # use iso ratio file?
    if ISO_RATIO == True:
        iso_flag = True
    else:
        iso_flag = False

    # define path and name of iso ratio file
    IsoTableFileName = IsoFileName

    # define path and name of file describing Non-LTE parameters
    CollisionFileName = ""

    # define number of pixels in x-direction (used for sub-beam description)
    NumModelPixelXX = n_points

    # define number of pixels in y-direction (used for sub-beam description)
    NumModelPixelYY = n_points

    # take local-overlap into account or not
    LocalOverlapFlag = False

    # disable sub-beam description
    NoSubBeamFlag = True

    # define path and name of database file
    dbFilename = "/scratch/beck/XCLASS/XCLASS__version_1.4.1/Database/cdms_sqlite.db"

    # define rest freq. (in MHz)
    RestFreq = 220709.01650

    # define v_lsr (in km/s)
    vLSR = 0.0
    
    warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### Mass of core, luminosity and L/M ###############################

    def get_density_integrand(r, rho, power_density):
        # Calculate the characteristic radius in centimeters
        r0 = Radius * pc_in_cm
        
        # Calculate the density value using the provided power-law profile
        density_value = rho * (r/r0)**-power_density
        
        # Ensure density_value is set to constant rho within the core radius
        if r < r0:
            density_value = rho
        
        # Return the integrand value: density times r squared
        return density_value * r * r

    # Define the integration limits
    lowerlim = 0
    upperlim = radius_core * pc_in_cm

    # Calculate the total number of CH3CN molecules within the core
    total_n_ch3cn = integrate.quad(get_density_integrand, lowerlim, upperlim, args=(rho, power_density))[0] * 4. * np.pi

    # Calculate the total number of H2 molecules within the core
    total_n_h2 = total_n_ch3cn / x_ch3cn

    # Convert the total mass of H2 molecules to kilograms
    total_m_kg = total_n_h2 * 2.33 * 1.6735575e-27

    # Calculate the core mass in terms of solar masses
    M_core = total_m_kg/M_Sun

    # Calculate the luminosity of the core using the Stefan-Boltzmann law
    L_core = (4*np.pi * sb * ((Radius * pc_in_m)**2) * (temperature**4))/L_sun

    # Calculate the ratio of luminosity to mass (L/M) for the core
    L_M_ratio = L_core/M_core

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### Making 3D artificial sources ###############################
        
    # these are the x, y-, and z- coordinates for the cubes that are created in the following steps
    x = np.linspace(-radius, radius, n_points)                                 
    y = np.linspace(-radius, radius, n_points)
    z = np.linspace(-radius, radius, n_points)

    
    # making a 3-dim array, where every point of the source is represented by a x-, y- and z- coordinate
    X, Y, Z = np.meshgrid(x, y, z)

    # pythagoras - getting my radius-points
    radius_mesh_3D = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)  # now I have a meshgrid which represents a distance

    ## Velocity-Cube
    radius_vector = np.stack([X, Y, Z], axis=-1)
    r = np.sqrt(np.sum(radius_vector ** 2, axis=-1)) # cube that represents the radius 

    # Calculate the velocity vector for every point in the grid
    velocity_vector = np.sqrt((2* mass_protostar * M_Sun * G) / (r* pc_in_m)**3)[:, :, :, np.newaxis] * radius_vector * pc_in_m

    index_Radius = np.argmin(np.abs(x - Radius)) # find one of the indices of x where the little radius lies, where the particles do not accelerate anymore
    index_center = n_points//2 # the index of the center on the x axis
    Radius_pixel = index_Radius - index_center # the radius of the little radius, where the particles do not accelerate anymore, in pixels              

    x_indices = np.arange(n_points)
    y_indices = np.arange(n_points)
    z_indices = np.arange(n_points)

    X_indices, Y_indices, Z_indices = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij') # A cube that represents all indices of the voxels in the cube

    mask = ((X_indices - index_center)**2 + (Y_indices - index_center)**2 + (Z_indices - index_center)**2) <= Radius_pixel**2 # Here I am making a mask which gives the indices of all voxels in the sphere that lay within the little radius

    velocity_magnitudes = np.linalg.norm(velocity_vector, axis=-1) # Compute the magnitudes of the velocity vectors
    unity_velocity_vector = np.zeros_like(velocity_vector) # Create an array to store the unity vectors
    unity_velocity_vector[mask] = velocity_vector[mask] / velocity_magnitudes[mask, np.newaxis] # Normalize the velocity vectors within the sphere and store them in the unity_velocity_vector array

    V_vector = velocity_vector[index_Radius, index_center, index_center] # Get the veclocity vector at the little radius on the line of sight
    V = np.abs(V_vector[1]) # This is the velocity that the particles have at the little radius "Radius"

    unity_velocity_vector[mask] = V * unity_velocity_vector[mask] # Multiplying the velocity with the unity vectors in the sphere 
    velocity_vector[mask] = unity_velocity_vector[mask] # Transferring these into the velocity_vector-Cube

    V_grid_3D = velocity_vector[:, :, :, 1] #getting the z components of the velocity-vectors 
    V_grid_3D[index_center, :, :] = 0 # particles should definitely be 0 here
    V_grid_3D[index_center, index_center, index_center] = np.nan # in the center the velocity should be a nan-value

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### Temperature calculation ###############################
    

    def get_temperature(x, y, z):
        ## input in parsecs ##
        
        # Calculate the distance from the origin for each point in space
        radius = np.sqrt(x**2 + y**2 + z**2)

        # Calculate the temperature values based on the given formula
        temperature_value = temperature * ((radius / (Radius))** -power_temperature)
        
        # Set temperature_value to constant temperature within the core radius
        if np.sqrt(x**2 + y**2 + z**2) < Radius: 
            temperature_value = temperature

        return temperature_value # in Kelvin
            
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### New density calculation ###############################

    def get_density(x, y, z):
        ## input in parsecs ##
        
        # Calculate the distance from the origin for each point in space
        radius = np.sqrt(x**2 + y**2 + z**2)

        # Calculate the density values based on the given formula
        density_value = rho * ((radius / (Radius * pc_in_cm))** -power_density)
        
        # Set density_value to constant rho within the core radius
        if np.sqrt(x**2 + y**2 + z**2) < Radius * pc_in_cm: 
            density_value = rho

        return density_value # in 1/cm^3

    
    
    def columndensity(Func, x_min, x_max, nx, y_min, y_max, ny, z_min, z_max, nz):

        ## integrates function Func(x,y,z) in the cuboid [ax,bx] * [ay,by] * [az,bz] using the trapezoidal rule with (nx * ny * nz) integration points
        ## taken from https://books.google.de/books?id=oCVZBAAAQBAJ&pg=PA390&lpg=PA390&dq=trapezoidal+rule+in+3d+python&source=bl&ots
        ##              =qDxRaL-fmt&sig=KbSEJ_tTzFgrvv_1UpYSZQV9h3E&hl=en&sa=X&ved=0ahUKEwj8ktLp9MbUAhVQalAKHa7_AUAQ6AEIYDAJ#v=onepage&q
        ##              =trapezoidal%20rule%20in%203d%20python&f=false

        """
        input parameters:
        -----------------
            - Func:                     function which is integrated
            - x_min:                    lower limit along x-axis
            - x_max:                    upper limit along x-axis
            - nx:                       grid numbers along x-axis
            - y_min:                    lower limit along y-axis
            - y_max:                    upper limit along y-axis
            - ny:                       grid numbers along y-axis
            - z_min:                    lower limit along z-axis
            - z_max:                    upper limit along z-axis
            - nz:                       grid numbers along z-axis

        output parameters:
        -----------------
            - s:                        computed integral
        """

        ## Define step sizes for each axis
        hx = (x_max - x_min) / (nx - 1)
        hy = (y_max - y_min) / (ny - 1)
        hz = (z_max - z_min) / (nz - 1)
        
        #print('hx = %s' %hx)
        # print('hy = %s' %hy)
        #print('hz = %s' %hz)
        
        #print ("x_min = ", x_min)
        #print ("x_max = ", x_max)
        #print ("nx = ", nx)
        # print ("y_min = ", y_min)
        #print ("y_max = ", y_max)
        #print ("ny = ", ny)
        #print ("z_min = ", z_min)
        #print ("z_max = ", z_max)
        #print ("nz = ", nz)
    
        ## Initialize array to store results
        S = np.array([])    
        
        ## Iterate over each component (p, q) in the cuboid
        for p, q in itertools.product(range(3), range(components)):
            s = 0.0
        ## Trapezoidal rule integration along the x-axis
            for i in range(0, nx):
                x = x_min + i * hx 
                #print('x =', x)
                wx = (hx if i * (i + 1 - nx) else 0.5 * hx) 
                #print('wx =', wx)
                sx = 0.0
                ## Trapezoidal rule integration along the y-axis
                for j in range(0, ny):
                    y = y_min + j * hy 
                    #print('y =', y)
                    wy = (hy if j * (j + 1 - ny) else 0.5 * hy) 
                    # print('wy =', wy)
                    sy = 0.0           
                    ## Trapezoidal rule integration along the z-axis
                    for k in range(0, nz):
                        z = z_min + k * hz 
                        #print('z =', z)
                        wz = (hz if k * (k + 1 - nz) else 0.5 * hz)
                        #print('wz =', wz)                        
                        sy += wz[q]  * Func(x, y[p], z[q])
                        #print('sy =', sy)                        
                    sx += wy[p]  * sy
                s += wx * sx        
            S = np.append(S, s)
        ## Reshape the result array and scale by pixel width
        S = np.reshape(S, (3, components))
        S = S/(pixel_width_cm**2)
        return S

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################### Creating the Components ###############################

    # Calculate the thickness of one component in parsec
    thickness_components_pc = 2 * radius / components  

    # Calculate the distance of the center of each component from the center of the star in parsec
    center_components = x[0] + np.arange(0.5, components + 0.5, 1.0) * thickness_components_pc  

    # Get the indices of the elements in x that are closest to center_components
    differences = np.abs(x[:, np.newaxis] - center_components)
    comp_center_args = np.argsort(differences, axis=0)[0]

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Select the 3D grid data corresponding to the component centers
    V_grid_3D = V_grid_3D[comp_center_args, :, :]

    # Set a specific value in the center component to 0
    V_grid_3D[components // 2, index_center, index_center] = 0

    # Convert V_grid_3D to kilometers/s
    V_grid_3D = V_grid_3D / 1000        

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ############################### Defining the corner coordinates to calculate the column density for each component afterwards ##################################################

    # Define the pixels where the spectra will be calculated in centimeters
    list_i = np.array([0, n_points // 4, n_points // 2 - Radius_pixel])
    pix_points = (list_i + 0.5) * pixel_width_cm  

    # Convert the x, y, and z coordinates to centimeters
    xKart = 0 * pc_in_cm  
    yKart = (radius * pc_in_cm - pix_points)  
    zKart = x[comp_center_args] * pc_in_cm  

    ## Define corner coordinates of the current cell for column density calculation
    xmin = xKart - (pixel_width_cm * 0.5)  
    xmax = xKart + (pixel_width_cm * 0.5)  
    nx = 10                                                                        
    ymin = yKart - (pixel_width_cm * 0.5)  
    ymax = yKart + (pixel_width_cm * 0.5)  
    ny = 10                                                                            
    zmin = zKart - (thickness_components_pc * 0.5) * pc_in_cm  
    zmax = zKart + (thickness_components_pc * 0.5) * pc_in_cm  
    nz = 10                   

    # Calculate the column density for each component
    n_values = columndensity(get_density, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz) 

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ############################### Defining the corner coordinates to calculate the column density for each component afterwards ##################################################

    # Create an empty list to store the temperature values
    T_values = []

    # Loop over all combinations of y and z coordinates
    for y in yKart / pc_in_cm:
        for z in zKart / pc_in_cm:
            temperature_value = get_temperature(xKart / pc_in_cm, y, z)
            T_values.append(temperature_value)
            
    # Convert the temperature values to a numpy array and reshape it
    T_values = np.array(T_values)
    T_values = np.reshape(T_values, (len(yKart), len(zKart)))

 #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    ############################### Creating the molfits-file for each pixel ##################################################
    
    Spectrum_array = np.array([[]])  # This is an empty array that will be filled with the calculated spectra for each parameter-combination

    # Preparing the molfit-file
    for i, j in zip(list_i, np.arange(3)):
        
        # Define a line in the molfit file to repeat for each parameter combination
        line_to_repeat = "n   0.02     2.0        {}       y   10.000  1000.00  {}      y  5.000e+8   5.000e+20   {}      y   2.000   30.00  6.00       y   -10.000 10.000  {}    {}"

        # Define the contents of the molfit file template
        file_contents = '''% Number of molecules   1
        %
        % schema:
        %
        % name of molecule              number of components
        % fit:  low:  up:  source size [arcsec]:    fit:  low:  up:  T_rot [K]:    fit:  low:  up:  N_tot [cm-2]:    fit:  low:  up:  velocity width [km/s]: fit:  low:  up:  velocity offset [km/s]:
        %
        CH3CN;v=0;              {}

        '''
        file_contents = file_contents.format(components)  # Fill the template with the number of components
        
        # Fill the molfit file with the necessary values for each parameter combination
        for values in zip(T_values[j, :], map("{:e}".format, n_values[j, :]), V_grid_3D[:, n_points//2, i], range(components, 0, -1)):
            line = line_to_repeat.format(pixel_width_arcsec, *values)  # Format the line with the current parameter values
            file_contents += line + '\n'  # Append the formatted line to the file contents
            
        # Write the contents to the molfit file
        with open(MolfitsFileName, "w") as file:
            file.write(file_contents)
        
        # Read the contents of the molfit file
        with open(MolfitsFileName, "r") as file:
            file_contentsfile_contents = file.read()
        
        # Suppress output
        # sys.stdout = open('nul', 'w')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
########################### call myXCLASS function ##################################################################
    # Calculate the spectra with XCLASS
    modeldata, log, TransEnergies, IntOpt, JobDir = task_myXCLASS.myXCLASS(
                                                    FreqMin, FreqMax, FreqStep,
                                                    TelescopeSize, BMIN, BMAJ,
                                                    BPA, Inter_Flag, Redshift,
                                                    t_back_flag, tBack, tslope,
                                                    BackgroundFileName,
                                                    N_H, beta_dust, kappa_1300,
                                                    DustFileName, Te_ff, EM_ff,
                                                    kappa_sync, B_sync, p_sync,
                                                    l_sync, ContPhenFuncID,
                                                    ContPhenFuncParam1,
                                                    ContPhenFuncParam2,
                                                    ContPhenFuncParam3,
                                                    ContPhenFuncParam4,
                                                    ContPhenFuncParam5,
                                                    MolfitsFileName, iso_flag,
                                                    IsoTableFileName,
                                                    CollisionFileName,
                                                    NumModelPixelXX,
                                                    NumModelPixelYY,
                                                    LocalOverlapFlag,
                                                    NoSubBeamFlag,
                                                    dbFilename,
                                                    RestFreq, vLSR)

    # Remove JobDir file
    delete_JobDir = "rm -rf " + JobDir
    os.system(delete_JobDir)

    if j == 2:
        # Remove Molfit file
        delete_Molfit = "rm -rf " + MolfitsFileName
        os.system(delete_Molfit)

        # Remove IsoRatio file
        delete_IsoRatio = "rm -rf " + IsoTableFileName
        os.system(delete_IsoRatio)

    # Append the calculated spectra to Spectrum_array
    Spectrum_array = np.append(Spectrum_array, modeldata[:, 2])

    # Construct parameter_array based on conditions
    if not (ISO_RATIO and NOISE):
        parameter_array = np.array([temperature, power_temperature, rho, power_density, mass_protostar, M_core, L_core, L_M_ratio, radius_core, radius, Radius])    

    if ISO_RATIO and not NOISE:
        parameter_array = np.array([temperature, power_temperature, rho, power_density, mass_protostar, iso_ratio, M_core, L_core, L_M_ratio, radius_core, radius, Radius])    

    if not ISO_RATIO and NOISE:
        parameter_array = np.array([temperature, power_temperature, rho, power_density, mass_protostar, noise, M_core, L_core, L_M_ratio, radius_core, radius, Radius])    

    if ISO_RATIO and NOISE:
        parameter_array = np.array([temperature, power_temperature, rho, power_density, mass_protostar, iso_ratio, noise, M_core, L_core, L_M_ratio, radius_core, radius, Radius])    

    # Extract Frequency_array from modeldata
    Frequency_array = modeldata[:, 0]

    return Spectrum_array, parameter_array, Frequency_array

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
############################## Iterating over all combinations with multiprocessing ###########################################

if __name__ == "__main__":
    
    # Set the number of processes (number of available CPU cores)
    num_processes = 60

    # Multiprocessing to accelerate the generating
    Spectrum_list = []
    parameter_list = []

    with Pool(num_processes) as pool:
        try:
            # Iterate over all combinations using multiprocessing
            for result in tqdm(pool.imap_unordered(process_combination, combinations), total=len(combinations)):
                Spectrum_list.append(result[0])  # Append spectrum results
                parameter_list.append(result[1])  # Append parameter results
        except TypeError:
            # Ignore errors and continue with the code
            pass
        pool.close()
        pool.join()

    # Combine results into arrays
    dataset = np.vstack(Spectrum_list)
    parameterset = np.vstack(parameter_list)
    Frequency_array = result[2]  # Extract Frequency_array from the last result

    # Get current date and time in a specific format
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output directory path appending the date and time
    output_directory = f"../data/{current_datetime}/"

    # Ensure the directory exists, if not, create it
    os.makedirs(output_directory, exist_ok=True)

    # Parameter names based on conditions
    if not ISO_RATIO and not NOISE:
        parameter_names = ['temperature', 'temperature exponent', 'density', 'density exponent', 'mass protostar', 'mass core', 'luminosity core', 'L/M', 'core radius', 'source radius', 'reference radius']

    if ISO_RATIO and not NOISE:
        parameter_names = ['temperature', 'temperature exponent', 'density', 'density exponent', 'mass protostar', 'iso ratio', 'mass core', 'luminosity core', 'L/M', 'core radius', 'source radius', 'reference radius']

    if not ISO_RATIO and NOISE:
        parameter_names = ['temperature', 'temperature exponent', 'density', 'density exponent', 'mass protostar', 'noise', 'mass core', 'luminosity core', 'L/M', 'core radius', 'source radius', 'reference radius']

    if ISO_RATIO and NOISE:
        parameter_names = ['temperature', 'temperature exponent', 'density', 'density exponent', 'mass protostar', 'iso ratio', 'noise', 'mass core', 'luminosity core', 'L/M', 'core radius', 'source radius', 'reference radius']

    # Save your data
    np.savetxt(output_directory + "parameterset.dat", parameterset, delimiter='\t', header='\t'.join(parameter_names))
    np.savetxt(output_directory + "dataset.dat", dataset)

    # Command to delete directories and their contents
    command1 = f"rm -r {directory_path}/*"
    command2 = f"rm -r {molfit_path}/*"

    # Execute commands in the terminal
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)