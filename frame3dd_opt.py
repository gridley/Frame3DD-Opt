#!/usr/bin/env python3
# frame3dd_opt -- a frame optimization tool to wrap Frame3DD (frame3dd.sourceforge.net)
# This tool uses the differential evolution algorithm to optimize frame designs under both
# geometric constraints and stability constraints. Given a certain truss topology and node
# variables to optimize, this tool can find an optimized member sizing and node placement
# pattern for the truss. The tool can work in either 2D or 3D.
from abc import ABC, abstractmethod
from jinja2 import Template
import numpy as np
import random
import subprocess # for running Frame3DD
import os # required for making temporary output directories for each MPI proc to use
from mpi4py import MPI
import scipy.optimize

class Frame3DDOutput:
    ''' Responsible for parsing output from Frame3DD. This
    includes member lengths, and member compressions and tensions.
    Assumes you're not turning on shear effects in the frame analysis.
    '''

    # Section separators in Frame3DD output file
    element_data_header = 'F R A M E   E L E M E N T   D A T A'
    node_data_header = 'N O D E   D A T A'
    element_force_header = 'F R A M E   E L E M E N T   E N D   F O R C E S'
    reactions = 'R E A C T I O N S'

    def __init__(self, outfile_name):
        frame_element_data = []
        node_data = []
        frame_element_reactions = []

        # set some variables used for sizing frames
        self.yield_stress = None
        self.beam_geometry = None

        with open(outfile_name) as outfile:
            self.node_data = [x for x in self.read_node_section(outfile)]
            self.frame_element_data = [x for x in self.read_element_data_section(outfile)]
            self.frame_element_reactions = [x for x in self.read_element_forces(outfile)]

    @classmethod
    def read_node_section(cls, filehandle):
        '''
        Each time this is called, this yields the next x,y,z coordinate of a node.
        This assumes that node indices have been assigned in strictly increasing
        numerical order. If you do not follow that convention, this method is incorrect.
        '''
        on_section = False
        for line in filehandle:
            if on_section:
                if line.startswith(cls.element_data_header):
                    next(filehandle)
                    return
                else:
                    split_line = line.split()
                    yield (float(split_line[1]), float(split_line[2]), float(split_line[3]))
            elif line.startswith(cls.node_data_header):
                next(filehandle) # skip a line
                on_section = True
        raise Exception('Could not find nodes section in Frame3DD output file.')

    @classmethod
    def read_element_data_section(cls, filehandle):
        '''
        Reads element geometric data. This _must_ be run AFTER
        read_node_section, and will be incorrect otherwise.
        It yields the node indices and member area (in a tuple) on each call
        '''
        for line in filehandle:
            if line.startswith('  Neglect shear deformations.'):
                return
            split_line = line.split()
            yield (int(split_line[1]), int(split_line[2]), float(split_line[3]))

    @classmethod
    def read_element_forces(cls, filehandle):
        '''
        returns element forces. Negative => compression,
        positive => tension. yields:

        (member ID, force)
        '''
        on_section = False
        for line in filehandle:
            if on_section:
                if line.startswith(cls.reactions):
                    return
                else:
                    split_line = line.split()
                    force = abs(float(split_line[2][:-1]))
                    if split_line[2][-1] == 'c':
                        force *= -1
                    yield (int(split_line[0]), force)
                    next(filehandle)
            elif line.startswith(cls.element_force_header):
                next(filehandle)
                on_section = True

    def calculate_frame_volume(self):
        ''' Calculates total volume of the structure.
        This willl often be a figure of merit for an optimization.
        '''
        total_volume = 0
        for member in self.frame_element_data:
            node1 = member[0] - 1 # convert to one-based indexing
            node2 = member[1] - 1
            member_length = 0
            for i in range(3):
                member_length += (self.node_data[node1][i]-self.node_data[node2][i])**2
            member_length = member_length**0.5
            total_volume += member_length * member[2]
        return total_volume

    def member_length(self, i):
        '''
        Calculates the length of the i_th member
        '''
        member = self.frame_element_data[i]
        node1 = member[0] - 1 # convert to one-based indexing
        node2 = member[1] - 1
        member_length = 0
        for j in range(3):
            member_length += (self.node_data[node1][j]-self.node_data[node2][j])**2
        member_length = member_length**0.5
        return member_length

    def calculate_f_times_l(self):
        ''' Calculates the common objective sum_i |F_i| L_i, which is directly related
        to the total mass of the truss as shown in MIT 4.450, or simple algebra. this assumes
        no buckling, however.
        '''
        sum_fl = 0
        for i, member in enumerate(self.frame_element_data):
            node1 = member[0] - 1 # convert to one-based indexing
            node2 = member[1] - 1
            member_length = 0
            for j in range(3):
                member_length += (self.node_data[node1][j]-self.node_data[node2][j])**2
            member_length = member_length**0.5
            sum_fl += member_length * abs(self.frame_element_reactions[i][1])
        return sum_fl

# Here, a two possible element shapes are defined: either round tubes or square tubes.
# This could be easily be modified to include other geometries in the future.
# The role of these are to:
# 1) Store a geometric parameter independent of area, for these two, that being the
#    relative size of the tube wall compared to its thickness.
# 2) Calculate torsional constant, and moments of area in the axes perpendicular to the member (for buckling)
class MemberGeometricPropertyCalculator(ABC):

    def __init__(self, area):
        self.area = area

    @abstractmethod
    def get_Asy(self):
        ''' Calculates the shear area in first perpendicular axis
        '''
        raise NotImplementedError
    @abstractmethod
    def get_Asz(self):
        ''' Calculates the shear area in first perpendicular axis
        '''
        raise NotImplementedError
    @abstractmethod
    def get_Jxx(self):
        ''' Calculates the torsional geometric constant of the member
        '''
        raise NotImplementedError
    @abstractmethod
    def get_Iyy(self):
        ''' Calculates the moment of are along the first perpendicular axis
        '''
        raise NotImplementedError
    @abstractmethod
    def get_Izz(self):
        ''' Calculates the moment of are along the second perpendicular axis
        '''
        raise NotImplementedError

class SquareTube(MemberGeometricPropertyCalculator):
    ''' given the ratio of the tube thickness
    to the side length, calculate the geometric properties.
    The thickness ratio can be set to 0.5 to have a solid tube.
    '''
    def __init__(self, area, thickness_to_side_ratio):
        super().__init__(area)
        self.t_over_s = thickness_to_side_ratio
        self.b = (area / (4 * (thickness_to_side_ratio - thickness_to_side_ratio**2)))**0.5
        self.t = self.t_over_s * self.b
    def get_Asy(self):
        return self.area / (2.39573 - 0.25009*self.t_over_s - 7.89675*self.t_over_s**2)
    def get_Asz(self):
        return self.get_Asy()
    def get_Jxx(self):
        return (self.b-self.t)**3 * self.t
    def get_Iyy(self):
        return (1/12) * (self.b**4 - (self.b - 2*self.t)**4)
    def get_Izz(self):
        return self.get_Iyy()

class RoundTube(MemberGeometricPropertyCalculator):
    ''' given the ratio of the tube radii and area, calculates properties
    '''
    def __init__(self, area, ri_over_ro):
        super().__init__(area)
        self.ri_over_ro = ri_over_ro
        self.ro = np.sqrt(self.area / (np.pi * (1-self.ri_over_ro**2)))
        self.ri = self.ri_over_ro * self.ro
    def get_Asy(self):
        return self.area / (0.54414 + 2.97294 * self.ri_over_ro - 1.51899 * self.ri_over_ro**2)
    def get_Asz(self):
        return self.get_Asy()
    def get_Jxx(self):
        return 0.5 * np.pi * (self.ro**4 - self.ri**4)
    def get_Iyy(self):
        return self.get_Jxx() / 2
    def get_Izz(self):
        return self.get_Iyy()

class UniformPerturbationSampler:
    '''
    Returns perturbations in the range [-pert_size, +pert_size], uniformly distributed
    '''
    def __init__(self, pert_size):
        self.pert_size = pert_size
    def __call__(self):
        return random.uniform(-pert_size, pert_size)

# A few material classes. Units are kips/inches (yikes)
class A36Steel:
    E = 29000 # elastic modulus (ksi)
    G = 11500 # shear modulus (ksi)
    tensile_yield_stress = 36.3 # ksi (typically 0.2% elongation)
    compressive_yield_stress = 22 # ksi (typically 0.2% elongation)
    density = 2.8e-4 # ksi/in
    expansion = 0 # not touching this for now

# A few material classes. Units are kips/inches (yikes)
class DouglasFir:
    E = 12400 # elastic modulus (ksi)
    G = 4600 # shear modulus (ksi)
    tensile_yield_stress = 5 # ksi (typically 0.2% elongation)
    compressive_yield_stress = 3.94 # ksi (typically 0.2% elongation)
    density = 2.2e-5 # ksi/in
    expansion = 0 # not touching this for now

class OptimizationProblem:
    '''
    Structural optimizations problems, in this code, are defined by a few things.
    Firstly, a valid, templated, input file to Frame3DD should be provided, with
    templated values filled in using Jinja2 syntax. Secondly, those variables must
    be listed separately, along with initial guesses and rules for perturbing those
    variables. Lastly, constraints on each variable must also be given. The final
    argument should be a list of 2-tuples of the length of the variable list which
    denote the allowable upper and lower bounds of each variable.

    With all that information, this class can run differential evolution on the
    problem, as it is defined. A few other solver settings are available.

    The Jinja2 template / Frame3DD input you provide MUST include a template variable
    where the geometric stiffness option is toggled. This template variable should
    be named {{ geom }}, and appears as the second entry after the frame element
    specifications.
    '''
    def __init__(self, template_file_name, connectivity_file, variable_file, safety_factor=2, population_per_rank=15, maxiter=1000,
            mutation=0.5, recombination=0.7, consider_local_buckling=True, consider_global_buckling=False, material=A36Steel, xmin=-1e8, xmax=1e8):

        # Load the Jinja2 template data
        with open(template_file_name, 'r') as template_fh:
            self.template = Template(template_fh.read())

        # Load node-to-node connectivities
        self.connectivities = np.loadtxt(connectivity_file, dtype=np.int32)
        self.n_members = self.connectivities.shape[0]
        self.member_thicknesses = np.zeros(self.n_members)
        if self.connectivities.shape[1] != 2:
            raise Exception('connectivities should be two entries per line')

        # Load in the variables to optimize on
        self.variable_names = []
        self.constrained_boundaries= []
        with open(variable_file, 'r') as variables_fh:
            for line in variables_fh:
                split_line = line.split()
                if len(split_line) != 3:
                    raise Exception("there should be three entries per line in the variables file")
                self.variable_names.append(split_line[0])
                assert float(split_line[1]) < float(split_line[2])
                self.constrained_boundaries.append((float(split_line[1]), float(split_line[2])))
        self.n_variables = len(self.variable_names)

        # Bounding box on allowable node positions. These keep them
        # from running off to infinity, which they do sometimes, annoyingly
        # if a design can be found putting zero force on that member.
        self.xmin = xmin
        self.xmax = xmax

        # Save various settings
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_size = MPI.COMM_WORLD.Get_size()
        self.safety_factor = safety_factor
        self.population_per_rank = population_per_rank
        self.population_size = self.population_per_rank * self.mpi_size
        self.constrain_global_buckling = consider_global_buckling
        self.maxiter = maxiter
        self.mutation = 0.5
        self.recombination = 0.7
        self.material = material
        self.consider_local_buckling = consider_local_buckling

        if not consider_local_buckling and not consider_global_buckling:
            self.evaluate_objective = self.evaluate_objective_force_times_length
        elif not consider_global_buckling:
            self.evaluate_objective = self.evaluate_objective_local_buckling
        elif consider_global_buckling:
            raise NotImplementedError('Have not added global buckling optimizations yet.')
        else:
            raise Exception('wtf??')

        # Allocate space for both previous iteration active populations
        # and last iteration active populations.
        self.last_iteration_population = np.zeros((self.population_size, self.n_variables))
        self.current_population = np.zeros_like(self.last_iteration_population)

        # Array of cost functions for the current population (just start at large value in case i mess up)
        self.cost_function = np.ones(self.population_size) * 1e100

        # Check that either the author or user of this code isn't insane
        if consider_global_buckling:
            raise NotImplementedError
        if not consider_local_buckling and consider_global_buckling:
            raise Exception('Cannot consider global buckling without local buckling.')

        # Ensure that temporary output directories are available for each proc to use
        if not os.path.exists('out%i'%self.rank):
            os.mkdir('out%i'%self.rank)

        # Get path for each proc to use
        self.path = os.getenv('PATH')

    def evaluate_objective_force_times_length(self, variable_values):
        '''
        Does a linear elasticity calculation, using large member thicknesses
        in order to calculate the sums of magnitudes of member forces times
        their lengths, which is proportional to frame mass under some simple
        assumptions.
        '''
        # Do elastic frame calculation to calculate member sizes
        inputname = 'input_%i.3dd'%self.rank
        outputname = 'output_%i.out'%self.rank
        member_string = '%i\n'%self.n_members
        for member_id in range(self.connectivities.shape[0]):
            member_string += '%i %i %i 1000.0 1.0 1.0 1.0 1.0 2 %f %f 0 %f\n'%(member_id+1, self.connectivities[member_id, 0],
                    self.connectivities[member_id, 1], self.material.E, self.material.G, self.material.density)
        with open(inputname, 'w') as fh:
            fh.write(self.template.render(members=member_string, nmodes=0, **dict(zip(self.variable_names, variable_values))))

        the_env = {'PATH':self.path, 'FRAME3DD_OUTDIR':'out%i'%self.rank}
        result = subprocess.run(['frame3dd', '-i', inputname, '-o', outputname, '-q'], env=the_env)

        # Note: exit code 182 is given for large strains. For the linear elastic analysis, I just set beam thicknesses to an arb. number,
        # so this is expected, and should not cause any difference in the solution.
        if result.returncode and result.returncode!=182:
            print("Frame3DD exited with error code %i on proc %i"%(result.returncode, self.rank))
            MPI.COMM_WORLD.Abort()

        # Read in results
        elastic_result = Frame3DDOutput(outputname)

        # First get an estimate on member sizes based on the yield stress
        total_frame_volume = 0
        for i in range(self.n_members):

            # Calculate area of member based on yield stress approach
            if elastic_result.frame_element_reactions[i][1] < 0:
                member_area = abs(elastic_result.frame_element_reactions[i][1]) / self.material.compressive_yield_stress * self.safety_factor
            else:
                member_area = abs(elastic_result.frame_element_reactions[i][1]) / self.material.tensile_yield_stress * self.safety_factor
            self.member_thicknesses[i] = member_area
            total_frame_volume += elastic_result.member_length(i) * member_area

        # Need to clean up the result, since Frame3DD just writes to make output files longer
        removal_result = subprocess.run(['rm', outputname])
        if removal_result.returncode:
            print("Frame3DD screwed up in some undecipherable way on proc %i..."%self.rank)
            MPI.COMM_WORLD.Abort()

        return total_frame_volume

    def evaluate_objective_local_buckling(self, variable_values):
        '''
        This proceeds in two steps. Firstly, a linear elasticity calculation is done in order to calculate the
        size of the members such that both local buckling, tensile yielding, and compressive yielding are all
        avoided. After that, a more accurate calculation is done which includes global buckling effects and geometric
        stiffness effects. If local buckling or global buckling are toggled to not be considered, the flow of this
        is a little bit different.
        '''
        # Do elastic frame calculation to calculate member sizes
        inputname = 'input_%i.3dd'%self.rank
        outputname = 'output_%i.out'%self.rank
        member_string = '%i\n'%self.n_members
        for member_id in range(self.connectivities.shape[0]):
            member_string += '%i %i %i 1000.0 1.0 1.0 1.0 1.0 2 %f %f 0 %f\n'%(member_id+1, self.connectivities[member_id, 0],
                    self.connectivities[member_id, 1], self.material.E, self.material.G, self.material.density)
        with open(inputname, 'w') as fh:
            fh.write(self.template.render(members=member_string, nmodes=0, **dict(zip(self.variable_names, variable_values))))

        the_env = {'PATH':self.path, 'FRAME3DD_OUTDIR':'out%i'%self.rank}
        result = subprocess.run(['frame3dd', '-i', inputname, '-o', outputname, '-q'], env=the_env)

        # Note: exit code 182 is given for large strains. For the linear elastic analysis, I just set beam thicknesses to an arb. number,
        # so this is expected, and should not cause any difference in the solution.
        if result.returncode and result.returncode!=182:
            print("Frame3DD exited with error code %i on proc %i"%(result.returncode, self.rank))
            MPI.COMM_WORLD.Abort()

        # Read in results
        elastic_result = Frame3DDOutput(outputname)

        total_frame_volume = 0

        # First get an estimate on member sizes based on the yield stress
        for i in range(self.n_members):

            # Calculate area of member based on yield stress approach
            if elastic_result.frame_element_reactions[i][1] > 0:
                member_area = elastic_result.frame_element_reactions[i][1] / self.material.tensile_yield_stress * self.safety_factor

            # If member is in compression, also do a buckling sizing:
            else:
                # First, size based off compressive yield stress
                member_area = abs(elastic_result.frame_element_reactions[i][1]) / self.material.compressive_yield_stress * self.safety_factor

                buckling_I = abs(elastic_result.frame_element_reactions[i][1]) * (0.65 * elastic_result.member_length(i))**2 * self.safety_factor / np.pi**2 / self.material.E

                # Now, given the minimum acceptable moment of area, the member area is calculated from that
                # under whatever geometric constraints have been placed on it. Hardcoded for circular tubes
                # of a certain aspect ratio for now.
                alpha = 0.9
                A = np.sqrt(4 * buckling_I * np.pi * (1-alpha**2)**2 / (1-alpha**4))
                if A > member_area:
                    member_area = A

            self.member_thicknesses[i] = member_area
            total_frame_volume += elastic_result.member_length(i) * member_area

        # Need to clean up the result, since Frame3DD just writes to make output files longer
        removal_result = subprocess.run(['rm', outputname])
        if removal_result.returncode:
            print("Frame3DD screwed up in some undecipherable way on proc %i..."%self.rank)
            MPI.COMM_WORLD.Abort()

        return total_frame_volume

    def write_input_with_sized_members(self, variable_values):
        '''
        Using the stored member areas from a linear elasticity calculation (self.member_thicknesses),
        this method writes a Frame3DD input file that uses those areas rather than a large arbitrary
        area. This input file is then recommended for use to check whether global buckling is taking
        place by turning on the geometric stiffness option and checking that the stiffness matrix
        remains positive definite.
        '''
        # Do elastic frame calculation to calculate member sizes
        inputname = 'input_%i.3dd'%self.rank
        outputname = 'output_%i.out'%self.rank
        member_string = '%i\n'%self.n_members
        for member_id in range(self.connectivities.shape[0]):
            area = self.member_thicknesses[member_id]
            geom = RoundTube(area, 0.9)
            member_string += '%i %i %i %f %f %f %f %f %f %f %f 0 %f\n'%(member_id+1, self.connectivities[member_id, 0],
                    self.connectivities[member_id, 1],area,
                    geom.get_Asy(),
                    geom.get_Asz(),
                    geom.get_Jxx(),
                    geom.get_Iyy(),
                    geom.get_Izz(),
                    self.material.E, self.material.G, self.material.density)
        with open(inputname, 'w') as fh:
            fh.write(self.template.render(members=member_string, nmodes=0, **dict(zip(self.variable_names, variable_values))))

    def optimize_scipy(self):
        '''
        Carries out differential evolution using a Scipy backend.
        '''
        result = scipy.optimize.differential_evolution(self.evaluate_objective, self.constrained_boundaries, disp=True)
        print(result)

    def generate_feasible_solution(self):

        # First off, create an initial guess array of feasible solutions
        for row_i in range(self.population_per_rank):

            # Sample a feasible solution. There is a max number of attempts
            state_is_feasible = False
            max_attempts = 1000 # max number of attempts to sample a feasible solution
            feasibility_attempt = 0 # iteration counter
            while not state_is_feasible:
                for j_var in range(self.n_variables):
                    self.current_population[self.rank * self.population_per_rank + row_i, j_var] = random.uniform(self.constrained_boundaries[j_var][0], self.constrained_boundaries[j_var][1])
                state_is_feasible = True # TODO replace with global buckling check
                self.cost_function[self.rank * self.population_per_rank + row_i] = self.evaluate_objective(self.current_population[self.rank * self.population_per_rank + row_i])
                feasibility_attempt += 1
                if feasibility_attempt > max_attempts:
                    raise Exception("Unable to sample a feasible solution. Either change problem specification or increase max_attempts in frame3dd_opt.py")

        self.synchronize_population_across_ranks()
        self.last_iteration_population = self.current_population


    def exclusive_sample(self, *args):
        '''
        Samples an integer in [0, population_size) which is guaranteed not equal to any
        of the provided arguments.
        '''
        while True:
            propose = random.randrange(0, self.population_size)
            if all([propose != a for a in args]):
                return propose

    def optimize_mine(self, print_convergence_history=True):
        '''
        Carries out differential evolution using my backend. Parallelized over MPI, with
        a fixed amount of population on each MPI rank.
        '''
        self.generate_feasible_solution()

        evolution_iteration = 0

        trial = np.zeros(self.n_variables) # work space for generating trial mutation

        if print_convergence_history and self.rank == 0:
            convergence_file = open('convergence.txt', 'w')
            convergence_file.write('# Iteration  Min objective  Max objective  Mean objective  St. Dev. objective\n')

        while evolution_iteration < self.maxiter:

            # Parallel loop over population (only loop over chunk of population owned by this MPI rank)
            for p in range(self.rank * self.population_per_rank, (self.rank+1) * self.population_per_rank):

                # Original Storn and Price method:
                # Get partners
                in_allowable_region = False
                while not in_allowable_region:
                    a = self.exclusive_sample(p)
                    b = self.exclusive_sample(p, a)
                    c = self.exclusive_sample(p, a, b)

                    j = random.randrange(0, self.n_variables)
                    for k in range(self.n_variables):
                        if random.random() < self.recombination or k == self.n_variables-1:
                            trial[j] = self.last_iteration_population[c][j] + self.mutation * (
                                    self.last_iteration_population[a][j] - self.last_iteration_population[b][j])
                        else:
                            trial[j] = self.last_iteration_population[p][j]
                        j = (j+1)%self.n_variables
                    if np.all(trial > self.xmin) and np.all(trial < self.xmax):
                        in_allowable_region = True
                # ---------------------------------

                # # Best1Bin
                # # Get partners
                # a = np.argmin(self.cost_function)
                # b = self.exclusive_sample(p, a)
                # c = self.exclusive_sample(p, a, b)

                # j = random.randrange(0, self.n_variables)
                # for k in range(self.n_variables):
                #     if random.random() < self.recombination or k == self.n_variables-1:
                #         trial[j] = self.last_iteration_population[a][j] + self.mutation * (
                #                 self.last_iteration_population[b][j] - self.last_iteration_population[c][j])
                #     else:
                #         trial[j] = self.last_iteration_population[p][j]
                #     j = (j+1)%self.n_variables
                # ---------------------------------

                score = self.evaluate_objective(trial)
                if score <= self.cost_function[p]:
                    self.current_population[p, :] = trial
                    self.cost_function[p] = score
                else:
                    self.current_population[p, :] = self.last_iteration_population[p, :]

            self.synchronize_population_across_ranks()
            self.last_iteration_population = self.current_population

            if (self.rank == 0):
                print("Iteration %i: f(x) = %f" % (evolution_iteration, np.min(self.cost_function)))
                if print_convergence_history:
                    convergence_file.write('%i %f %f %f %f\n'%(evolution_iteration, np.min(self.cost_function),
                        np.max(self.cost_function), np.mean(self.cost_function), np.std(self.cost_function)))

            evolution_iteration += 1

        # Make it so that the winning one definitely has output printed out showing its design
        if self.rank == 0:
            best_design = np.argmin(self.cost_function)


            # Now polish off the solution using a gradient-based method
            self.evaluate_objective(self.current_population[best_design, :])
            print('polishing solution with CG...')
            result = scipy.optimize.minimize(self.evaluate_objective, x0=self.current_population[best_design, :], method='CG',
                    options={'gtol': .01, 'norm': 2, 'eps': .01, 'maxiter': 20, 'disp': True, 'return_all': False, 'finite_diff_rel_step': None})
            print(result)
            self.write_input_with_sized_members(result.x)
            np.savetxt('member_thicknesses', self.member_thicknesses)

    def synchronize_population_across_ranks(self):
        '''
        Synchronizes the modified populations across all ranks
        '''
        MPI.COMM_WORLD.Barrier()
        for rank in range(self.mpi_size):
            indx_start = rank * self.population_per_rank
            MPI.COMM_WORLD.Bcast(self.current_population[indx_start:indx_start+self.population_per_rank], root=rank)
            MPI.COMM_WORLD.Bcast(self.cost_function[indx_start:indx_start+self.population_per_rank], root=rank)
        MPI.COMM_WORLD.Barrier()

if __name__ == '__main__':
    import argparse
    # TODO add check that frame3dd is indeed present on the system.

    # When a new material class is added, put it in this dictionary to make it available on the command line
    material_map = {'A36Steel': A36Steel, 'DouglasFir': DouglasFir}

    cmd_parser = argparse.ArgumentParser(description='Frame3DD-Opt: Parallel differential evolution of frame designs')
    cmd_parser.add_argument('frame3dd_template', help="name of the Frame3DD input file template, formatted as per this program's manual")
    cmd_parser.add_argument('connectivity_file', help="a file containing a member topology definition. One line per member, with two numbers being the 1-based node indices.")
    cmd_parser.add_argument('variable_file', help="this file contains rows of variables to optimize over in the Frame3DD template file.")
    cmd_parser.add_argument('--no_local_buckling', dest='local_buckling', action='store_false', help="turn off consideration of local buckling when sizing frame members")
    cmd_parser.add_argument('--no_global_buckling', dest='global_buckling',  action='store_false', help="turn off constraining the optimization to avoid global buckling")
    cmd_parser.add_argument('--safety_factor', type=float, default=5, help="safety factor to use when sizing members", metavar='')
    cmd_parser.add_argument('--mutation', type=float, default=0.5, help="mutation factor for differential evolution", metavar='')
    cmd_parser.add_argument('--recombination', type=float, default=0.7, help="recombination factor for differential evolution", metavar='')
    cmd_parser.add_argument('--material', default='A36Steel', help="material to use for frame members. available types:\n %s"%('    \n'.join(material_map.keys())), metavar='')
    cmd_parser.add_argument('--population_per_rank', type=int, default=15, help="differential evolution population size per MPI rank", metavar='')
    cmd_parser.add_argument('--max_iter', type=int, default=1000, help="max DE iterations", metavar='')
    cmd_parser.add_argument('--xmin', type=float, default=-1e8, help="min coordinate", metavar='')
    cmd_parser.add_argument('--xmax', type=float, default=1e8, help="max coordinate", metavar='')
    cmd_parser.set_defaults(local_buckling=True, global_buckling=True)
    args = cmd_parser.parse_args()

    opt = OptimizationProblem(args.frame3dd_template,
            args.connectivity_file,
            args.variable_file,
            consider_local_buckling=args.local_buckling,
            consider_global_buckling=args.global_buckling,
            safety_factor=args.safety_factor,
            mutation=args.mutation,
            recombination=args.recombination,
            material=material_map[args.material],
            population_per_rank=args.population_per_rank,
            maxiter=args.max_iter,
            xmin=args.xmin,
            xmax=args.xmax)

    # opt.optimize_scipy()
    opt.optimize_mine()
