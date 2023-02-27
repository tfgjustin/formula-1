import argparse
import threading
import _io

from copy import deepcopy


def is_path_arg(arg):
    """Should we use the value of this variable in the file path name if logfile uses parameters?"""
    return not (arg.endswith('_tsv') or arg.startswith('_') or arg.startswith('logfile') or arg.startswith('print')
                or arg.startswith('run'))


def create_base_path(args):
    if args.logfile_uses_parameters:
        return '-'.join([args.logfile] + [str(getattr(args, arg)) for arg in dir(args) if is_path_arg(arg)])
    else:
        return args.logfile


def print_args(args, filename):
    with open(filename, 'w') as outfile:
        for arg in dir(args):
            if not is_path_arg(arg):
                continue
            print('--%s=%s' % (arg, str(getattr(args, arg))), file=outfile)


def add_common_args(parser):
    """These args are common to both the ratings calculator and the future simulator.
    """
    parser.add_argument('--driver_elo_initial', help='Initial driver Elo rating.',
                        type=int, default=1400)
    parser.add_argument('--driver_elo_regress',
                        help='Driver new season Elo regression multiplier.',
                        type=float, default=0.15)
    parser.add_argument('--driver_kfactor_regress',
                        help='Driver new season K-Factor regression multiplier.',
                        type=float, default=0.05)
    parser.add_argument('--driver_reliability_decay',
                        help='Rate at which we decay old reliability data for drivers.',
                        type=float, default=0.98)
    parser.add_argument('--driver_reliability_failure_constant',
                        help='Number of "failure" KM we add on a crash when calculating the per-KM failure odds.',
                        type=float, default=0.80)
    parser.add_argument('--driver_reliability_lookback',
                        help='Lookback window in races for driver reliability data.',
                        type=int, default=128)
    parser.add_argument('--driver_reliability_regress',
                        help='Percent by which we shade driver reliability data to the field average each year.',
                        type=float, default=0.3)
    parser.add_argument('--elo_compare_window',
                        help='Compare two results only if their Elo scores are within this difference.',
                        type=float, default=2.0)
    parser.add_argument('--elo_exponent_denominator_qualifying',
                        help='The denominator in the Elo probability exponent for races.',
                        type=int, default=180)
    parser.add_argument('--elo_exponent_denominator_race',
                        help='The denominator in the Elo probability exponent for races.',
                        type=int, default=270)
    parser.add_argument('--logfile_uses_parameters',
                        help='Append encoded parameters to the logfile output name.',
                        default=False, action='store_true')
    parser.add_argument('--num_iterations',
                        help='The number of times to simulate an event outcome to estimate win and podium odds.',
                        type=int, default=20000)
    parser.add_argument('--qualifying_kfactor_multiplier',
                        help='Value by which to multiply K-Factors during qualifying.',
                        type=float, default=0.40)
    parser.add_argument('--position_base_spec', help='Base Elo boost per starting position advantage.',
                        type=str, default='20_0_0')
#                        type = calculator.validate_factors, default = '10_4_1')
    parser.add_argument('--position_base_factor',
                        help='Exponent for base Elo boost per starting position advantage.',
                        type=float, default=0.85)
    parser.add_argument('--print_args',
                        help='Print a file containing the command-line args used for this run.',
                        default=False, action='store_true')
    parser.add_argument('--print_debug',
                        help='Print a file containing detailed debugging information (this will be large).',
                        default=False, action='store_true')
    parser.add_argument('--print_predictions',
                        help='Print a file containing all the predictions, useful in calibrating the model.',
                        default=False, action='store_true')
    parser.add_argument('--print_progress',
                        help='Print a file logging the progress of the calculations.',
                        default=False, action='store_true')
    parser.add_argument('--reliability_km_multiplier_street',
                        help='Per-KM driver reliability multiplier for street races.',
                        type=float, default=0.9995)
    parser.add_argument('--reliability_km_multiplier_wet',
                        help='Per-KM driver reliability multiplier for wet races.',
                        type=float, default=0.99945)
    parser.add_argument('--run_index', help='Which run is this.',
                        type=int, default=1)
    parser.add_argument('--run_max', help='Maximum number of runs.',
                        type=int, default=1)
    parser.add_argument('--start_year', help='First year for which we calculate ratings.',
                        type=int, default=1996)
    parser.add_argument('--team_elo_initial', help='Initial team Elo rating.',
                        type=int, default=1200)
    parser.add_argument('--team_elo_regress',
                        help='Team new season Elo regression multiplier.',
                        type=float, default=0.15)
    parser.add_argument('--team_kfactor_regress',
                        help='Team new season K-Factor regression multiplier.',
                        type=float, default=0.10)
    parser.add_argument('--team_reliability_decay',
                        help='Rate at which we decay old reliability data for teams.',
                        type=float, default=0.995)
    parser.add_argument('--team_reliability_failure_constant',
                        help='Number of "failure" KM we add on a car failure when calculating the per-KM failure odds.',
                        type=float, default=0.56)
    parser.add_argument('--team_reliability_lookback',
                        help='Lookback window in races for team reliability data.',
                        type=int, default=48)
    parser.add_argument('--team_reliability_new_events',
                        help='Number of events during which a new team is still considered "new".',
                        type=int, default=16)
    parser.add_argument('--team_reliability_regress',
                        help='Percent by which we shade team reliability data to the field average each year.',
                        type=float, default=0.50)
    parser.add_argument('--team_share_spec',
                        help='Fraction of the combined Elo rating belonging to the team.',
                        type=str, default='65_0_0')
#                        type = calculator.validate_factors, default = '50_4_1')
    parser.add_argument('--teammate_kfactor_multiplier',
                        help='Factor by which we multiply teammate K-Factors to increase/decrease points swapped.',
                        type=float, default=0.85)
    parser.add_argument('--wear_reliability_percent',
                        help='Percent of the wear reliability ratio to use when calculating failure rates.',
                        type=float, default=0.638)
    parser.add_argument('--wet_multiplier_elo_denominator',
                        help='Amount by which we multiple the Elo denominator during a wet event.',
                        type=float, default=1.15)
    parser.add_argument('--wet_multiplier_k_factor',
                        help='Amount by which we multiple the K-Factor during a wet event.',
                        type=float, default=0.75)


def add_common_positional_args(parser):
    parser.add_argument('logfile',
                        help='Write results to this logfile.')
    parser.add_argument('drivers_tsv', help='TSV file with the list of drivers.',
                        type=argparse.FileType('r'))
    parser.add_argument('events_tsv', help='TSV file with the list of events.',
                        type=argparse.FileType('r'))
    parser.add_argument('results_tsv', help='TSV file with the list of results.',
                        type=argparse.FileType('r'))
    parser.add_argument('teams_tsv', help='TSV file with the history of F1 teams.',
                        type=argparse.FileType('r'))


def csv_int(argument):
    args = list()
    for a in argument.split(','):
        try:
            value = int(a)
            args.append(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Argument %s contains non-integer value %s' % (argument, a))
    return args


def csv_float(argument):
    args = list()
    for a in argument.split(','):
        try:
            value = float(a)
            args.append(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Argument %s contains non-float value %s' % (argument, a))
    return args


def csv_str(argument):
    args = list()
    for a in argument.split(','):
        try:
            value = str(a)
            args.append(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Argument %s contains non-string value %s' % (argument, a))
    return args


class ArgFactory(object):

    def __init__(self, parser_template):
        self._parser_template = parser_template
        self._parser = self.create_argparser()
        self._lock = threading.Lock()
        # Arguments parsed from the command line, in list form
        self._parsed_args = None
        # The full set of option names
        self._opt_names = None
        # Positional args, which won't use '--' as a prefix
        self._positional_opts = None
        # Options without args
        self._bare_opts = None
        # Keep track of where we are in looping over the set of combinations
        self._current_idx = None
        self._max_idx = None
        self._used_combinations = 0
        self._max_combinations = 1
        # Byte streams from loaded files
        self._file_contents = dict()

    def parser_template(self):
        return self._parser_template

    def parser(self):
        return self._parser

    def parse_args(self):
        """
        Parse the arguments and initiate the counters.
        """
        self._current_idx = dict()
        self._max_idx = dict()
        self._opt_names = list()
        self._positional_opts = set()
        self._bare_opts = set()
        self._parsed_args = self._parser.parse_args()
        self.load_inputs()
        num_combinations = 1
        for action in self._parser._actions:
            option = action.dest
            if len(action.option_strings) > 1:
                continue
            elif not len(action.option_strings):
                self._positional_opts.add(option)
            elif action.nargs is not None and action.nargs == 0:
                self._bare_opts.add(option)
            self._opt_names.append(option)
            self._current_idx[option] = 0
            if isinstance(action.default, list):
                self._max_idx[option] = len(vars(self._parsed_args).get(option))
            else:
                self._max_idx[option] = 1
            num_combinations *= self._max_idx[option]
        self._max_combinations = num_combinations
        self._used_combinations = 0

    def max_combinations(self):
        """
        Return the maximum number of combinations of args handed to us.
        """
        return self._max_combinations

    def load_inputs(self):
        """
        We assume anything which is of type _io.TextIOWrapper is a file we should read. Store the bytes.

        TODO: Handle outputs (i.e., write TextIOWrappers)
        """
        for action in self._parser._actions:
            if not hasattr(self._parsed_args, action.dest):
                continue
            values = getattr(self._parsed_args, action.dest)
            if values is not None and isinstance(values, _io.TextIOWrapper):
                self._file_contents[values.name] = values.read()

    def next_config(self):
        """
        Create the next configuration. Returns None if we're done.
        """
        self._lock.acquire()
        current_args = self.create_current_args()
        if current_args is not None:
            # We're not at the end, so increment the positional counters.
            self.increment_index()
        self._lock.release()
        return current_args

    def create_current_args(self):
        """
        Create the args array for the current run.
        """
        if self._used_combinations >= self._max_combinations:
            return None
        current_args = []
        for option in self._opt_names:
            values = getattr(self._parsed_args, option)
            current_idx = self._current_idx.get(option, 0)
            # Special-case run_index and run_max since we use those to let the called program know where it is in the
            # run progress. Useful for printing out counters.
            if option == 'run_index':
                current_args.extend(['--' + option, str(self._used_combinations + 1)])
            elif option == 'run_max':
                current_args.extend(['--' + option, str(self._max_combinations)])
            elif option in self._positional_opts:
                if isinstance(values, _io.TextIOWrapper):
                    # If this is a positional arg and is a TextIOWrapper (i.e., something we've read) instead of
                    # returning the path to the file or the file wrapper, return the bytes we read from that file.
                    file_contents = self._file_contents.get(values.name, '')
                    current_args.append(file_contents)
                else:
                    # This is just a positional arg but not a file. Return the arg as-is.
                    current_args.append(values)
            elif option in self._bare_opts:
                # This is a bare option (e.g., one which indicates a boolean value). Just pass it as-is.
                if values:
                    current_args.extend(['--' + option])
            elif not isinstance(values, list):
                # This is not an arg which we allow combinations of values. Just include the option value as-is.
                current_args.extend(['--' + option, str(values)])
            else:
                # This is a combinatorial arg. Pass along the appropriate one for this run.
                current_args.extend(['--' + option, str(values[current_idx])])
        # Current args now contains the array as it would look if it were passed along on the regular command line, with
        # the appropriate combinatorial args selected for this particular run. Parse and return.
        return self._parser_template.parse_args(current_args)

    def increment_index(self):
        """
        Increment the counter for each arg, overflowing into the next one as needed.

        For example, if the maximum number of args in each position is [2, 3, 1, 2] and we're currently at 0-index of
        [1, 2, 0, 0], incrementing it would return [0, 0, 0, 1]:
        idx0: Incrementing 1 to 2 would hit the max, so reset to 0 and carry over
        idx1: Incrementing 2 to 3 would hit the max, so reset to 0 and carry over
        idx2: Incrementing 0 to 1 would hit the max, so reset to 0 and carry over
        idx3: Incrementing 0 to 1 would NOT hit the max, so increment and return
        """
        self._used_combinations += 1
        for option in self._opt_names:
            if self._current_idx[option] + 1 < self._max_idx[option]:
                self._current_idx[option] += 1
                return
            else:
                self._current_idx[option] = 0

    def create_argparser(self):
        """
        Do deep magic in the parser actions.
        """
        p = deepcopy(self._parser_template)
        for idx in range(len(p._actions)):
            if p._actions[idx].type == int:
                p._actions[idx].type = csv_int
                p._actions[idx].default = [p._actions[idx].default]
            elif p._actions[idx].type == float:
                p._actions[idx].type = csv_float
                p._actions[idx].default = [p._actions[idx].default]
            elif p._actions[idx].type == str:
                p._actions[idx].type = csv_str
                p._actions[idx].default = [p._actions[idx].default]
            elif isinstance(p._actions[idx].type, argparse.FileType):
                # Swap out the FileType with the string that we're going to pass in.
                self._parser_template._actions[idx].type = str
        return p
