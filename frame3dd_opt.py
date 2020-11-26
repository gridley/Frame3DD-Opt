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
    yield_stress = 36.3 # ksi (typically 0.2% elongation)
    density = 7.33e-7 # ksi/in
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
    def __init__(self, template_file_name, connectivity_file, variable_file, safety_factor=5, population_per_rank=15, maxiter=1000,
            mutation=0.5, recombination=0.7, consider_local_buckling=True, consider_global_buckling=False, material=A36Steel):

        # Load the Jinja2 template data
        with open(template_file_name, 'r') as template_fh:
            self.template = Template(template_fh.read())

        # Load node-to-node connectivities
        self.connectivities = np.loadtxt(connectivity_file, dtype=np.int32)
        self.n_members = self.connectivities.shape[0]
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

        # Save various settings
        self.safety_factor = safety_factor
        self.population_per_rank = population_per_rank
        self.constrain_global_buckling = consider_global_buckling
        self.maxiter = 1000
        self.mutation = 0.5
        self.recombination = 0.7
        self.material = material

        # Save a few important things
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_size = MPI.COMM_WORLD.Get_size()

        # Allocate space for both previous iteration active populations
        # and last iteration active populations.
        self.last_iteration_population = np.zeros((self.mpi_size * population_per_rank, self.n_variables))
        self.current_population = np.zeros_like(self.last_iteration_population)

        # Check that either the author or user of this code isn't insane
        if consider_local_buckling:
            raise NotImplementedError
        if consider_global_buckling:
            raise NotImplementedError
        if not consider_local_buckling and consider_global_buckling:
            raise Exception('Cannot consider global buckling without local buckling.')

    def evaluate_objective(self, variable_values):
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
        result = subprocess.run(['frame3dd', '-i', inputname, '-o', outputname, '-q'])

        # Note: exit code 182 is given for large strains. For the linear elastic analysis, I just set beam thicknesses to an arb. number,
        # so this is expected, and should not cause any difference in the solution.
        if result.returncode and result.returncode!=182:
            raise Exception("Frame3DD exited with error code %i"%result.returncode)

        # Read in results
        elastic_result = Frame3DDOutput(outputname)

        # Need to clean up the result, since Frame3DD just writes to make output files longer
        removal_result = subprocess.run(['rm', outputname])
        if result.returncode:
            raise Exception("Frame3DD screwed up in some undecipherable way...")

        return elastic_result.calculate_f_times_l()

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
                feasibility_attempt += 1
                if feasibility_attempt > max_attempts:
                    raise Exception("Unable to sample a feasible solution. Either change problem specification or increase max_attempts in frame3dd_opt.py")

        self.synchronize_population_across_ranks()
        self.last_iteration_population = self.current_population

    def optimize_mine(self):
        '''
        Carries out differential evolution using my backend
        '''
        self.generate_feasible_solution()

    def synchronize_population_across_ranks(self):
        '''
        Synchronizes the modified populations across all ranks
        '''
        MPI.COMM_WORLD.Barrier()
        for rank in range(self.mpi_size):
            indx_start = rank * self.population_per_rank
            MPI.COMM_WORLD.Bcast(self.current_population[indx_start:indx_start+self.population_per_rank], root=rank)
            MPI.COMM_WORLD.Barrier()

if __name__ == '__main__':
    import argparse
    # TODO add check that frame3dd is indeed present on the system.

    # When a new material class is added, put it in this dictionary to make it available on the command line
    material_map = {'A36Steel': A36Steel}

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
            population_per_rank=args.population_per_rank)

    # opt.optimize_scipy()
    opt.optimize_mine()
