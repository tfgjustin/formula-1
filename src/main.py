import argparse
import calculator
import data_loader
import math
import sys

from args import ArgFactory, add_common_args, create_base_path, print_args
from parallel_worker import parallel_run


def create_argparser():
    parser = argparse.ArgumentParser(description='Formula 1 rating parameters', fromfile_prefix_chars='@')
    parser.add_argument('drivers_tsv', help='TSV file with the list of drivers.',
                        type=argparse.FileType('r'))
    parser.add_argument('events_tsv', help='TSV file with the list of events.',
                        type=argparse.FileType('r'))
    parser.add_argument('results_tsv', help='TSV file with the list of results.',
                        type=argparse.FileType('r'))
    parser.add_argument('teams_tsv', help='TSV file with the history of F1 teams.',
                        type=argparse.FileType('r'))
    parser.add_argument('logfile',
                        help='Write results to this logfile.')
    parser.add_argument('--print_future_simulations',
                        help='Print a file logging results of simulated future results.',
                        default=False, action='store_true')
    add_common_args(parser)
    return parser


def run_one_combination(args, task_queue):
    base_path = create_base_path(args)
    if args.print_args:
        print_args(args, base_path + '.args')
    precision = math.ceil(math.log10(args.run_max))
    print_str = '[ %%%dd / %%%dd ] %%s' % (precision, precision)
    print(print_str % (args.run_index, args.run_max, base_path))
    with data_loader.DataLoader(args, base_path) as loader:
        loader.load_events(args.events_tsv)
        loader.load_drivers(args.drivers_tsv)
        if not loader.load_teams(args.teams_tsv):
            print('ERROR during %s' % base_path)
            if task_queue is not None:
                task_queue.task_one()
            return True
        loader.load_results(args.results_tsv)
        rating_calculator = calculator.Calculator(args, base_path)
        rating_calculator.run_all_ratings(loader)
    if task_queue is not None:
        task_queue.task_done()
    return True


def main():
    factory = ArgFactory(create_argparser())
    factory.parse_args()
    print('Running %d combination(s)' % factory.max_combinations())
    parallel_run(factory, run_one_combination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
