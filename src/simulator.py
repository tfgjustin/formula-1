import argparse
import calculator
import data_loader
import math
import sys

from args import ArgFactory, add_common_args, create_base_path
from parallel_worker import parallel_run


def create_argparser():
    parser = argparse.ArgumentParser(description='Formula 1 simulator parameters', fromfile_prefix_chars='@')
    parser.add_argument('drivers_tsv', help='TSV file with the list of drivers.',
                        type=argparse.FileType('r'))
    parser.add_argument('driver_ratings_tsv', help='TSV file with the log of driver ratings.',
                        type=argparse.FileType('r'))
    parser.add_argument('team_ratings_tsv', help='TSV file with the log of team ratings.',
                        type=argparse.FileType('r'))
#    parser.add_argument('driver_aggregates_tsv', help='TSV file with the log of driver aggregates.',
#                        type=argparse.FileType('r'))
#    parser.add_argument('team_aggregates_tsv', help='TSV file with the log of team aggregates.',
#                        type=argparse.FileType('r'))
    parser.add_argument('events_tsv',
                        help='TSV file containing list of previous events.',
                        type=argparse.FileType('r'), default='')
    parser.add_argument('future_events_tsv',
                        help='TSV file containing list of future events.',
                        type=argparse.FileType('r'), default='')
    parser.add_argument('logfile',
                        help='Write results to this logfile.')
    parser.add_argument('--future_lineup_tsv',
                        help='TSV file containing driver and team lineup for future events.',
                        type=str, default='')
    parser.add_argument('--print_future_simulations',
                        help='Print a file logging results of simulated future results.',
                        default=True, action='store_true')
    add_common_args(parser)
    return parser


def run_one_combination(args, task_queue):
    base_path = create_base_path(args)
    precision = math.ceil(math.log10(args.run_max))
    print_str = '[ %%%dd / %%%dd ] %%s' % (precision, precision)
    print(print_str % (args.run_index, args.run_max, base_path))
    with data_loader.DataLoader(args, base_path) as loader:
        # Load the events
        loader.load_events(args.events_tsv)
        # Load the driver database
        loader.load_drivers(args.drivers_tsv)
        # Load historical driver data
        if not loader.update_drivers_from_ratings(args.driver_ratings_tsv):
            print('ERROR loading drivers from ratings during %s' % base_path)
            if task_queue is not None:
                task_queue.task_one()
            return True
        # Load historical team data
        if not loader.load_teams_from_ratings(args.team_ratings_tsv):
            print('ERROR loading teams from ratings during %s' % base_path)
            if task_queue is not None:
                task_queue.task_one()
            return True
        # TODO: Load driver aggregates for use in projecting next-season values
        # TODO: Load team aggregates for use in projecting next-season values
        # Load future simulation events and determine future lineups (file or last event)
        loader.load_future_simulation_data()
        rating_calculator = calculator.Calculator(args, base_path)
        # Simulate
        rating_calculator.simulate_future_events(loader)
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
