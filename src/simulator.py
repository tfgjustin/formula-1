import argparse
import args
import calculator
import data_loader
import math
import sys

from parallel_worker import parallel_run


def create_argparser():
    parser = argparse.ArgumentParser(description='Formula 1 simulator parameters', fromfile_prefix_chars='@')
    args.add_common_positional_args(parser)
    parser.add_argument('future_events_tsv',
                        help='TSV file containing list of future events.',
                        type=argparse.FileType('r'), default='')
    parser.add_argument('team_adjust_tsv', help='TSV file with per-team rating adjustments.',
                        type=argparse.FileType('r'))
    parser.add_argument('driver_ratings_tsv', help='TSV file with the log of driver ratings.',
                        type=argparse.FileType('r'))
    parser.add_argument('team_ratings_tsv', help='TSV file with the log of team ratings.',
                        type=argparse.FileType('r'))
    parser.add_argument('grid_penalties_tsv', help='TSV file with the log of grid_penalties.',
                        type=argparse.FileType('r'))
#    parser.add_argument('driver_aggregates_tsv', help='TSV file with the log of driver aggregates.',
#                        type=argparse.FileType('r'))
#    parser.add_argument('team_aggregates_tsv', help='TSV file with the log of team aggregates.',
#                        type=argparse.FileType('r'))
    parser.add_argument('--future_lineup_tsv',
                        help='TSV file containing driver and team lineup for future events.',
                        type=str, default='')
    parser.add_argument('--print_future_simulations',
                        help='Print a file logging results of simulated future results.',
                        default=True, action='store_true')
    parser.add_argument('--print_ratings',
                        help='Print files logging ratings for teams and drivers.',
                        default=False, action='store_true')
    parser.add_argument('--simulate_season',
                        help='Season to simulate.',
                        type=str, default='2023')
    parser.add_argument('--simulate_start_round',
                        help='The first round in the season we will simulate; if "XX" start with the final real event.',
                        type=str, default='XX')
    args.add_common_args(parser)
    return parser


def run_one_combination(parsed_args, task_queue):
    base_path = args.create_base_path(parsed_args)
    precision = math.ceil(math.log10(parsed_args.run_max))
    print_str = '[ %%%dd / %%%dd ] %%s' % (precision, precision)
    print(print_str % (parsed_args.run_index, parsed_args.run_max, base_path))
    if task_queue is not None:
        task_queue.task_done()
    # return True
    with data_loader.DataLoader(parsed_args, base_path) as loader:
        # Load the events
        loader.load_events(parsed_args.events_tsv)
        # Load the driver database
        loader.load_drivers(parsed_args.drivers_tsv)
        # Team history
        if not loader.load_teams(parsed_args.teams_tsv):
            print('ERROR during %s' % base_path)
            if task_queue is not None:
                task_queue.task_done()
            return True
        # Historical results
        loader.load_results(parsed_args.results_tsv)
        if not loader.load_grid_penalties(parsed_args.grid_penalties_tsv):
            print('ERROR loading grid penalties during %s' % base_path)
            if task_queue is not None:
                task_queue.task_done()
            return True
        # Load historical driver data
        if not loader.update_drivers_from_ratings(parsed_args.driver_ratings_tsv):
            print('ERROR loading drivers from ratings during %s' % base_path)
            if task_queue is not None:
                task_queue.task_done()
            return True
        # Load historical team data
        if not loader.load_teams_from_ratings(parsed_args.team_ratings_tsv):
            print('ERROR loading teams from ratings during %s' % base_path)
            if task_queue is not None:
                task_queue.task_done()
            return True
        # TODO: Load driver aggregates for use in projecting next-season values
        # TODO: Load team aggregates for use in projecting next-season values
        # Load future simulation events and determine future lineups (file or last event)
        loader.load_future_simulation_data()
        rating_calculator = calculator.Calculator(parsed_args, base_path)
        # Simulate
        rating_calculator.simulate_future_events(loader)
    if task_queue is not None:
        task_queue.task_done()
    return True


def main():
    factory = args.ArgFactory(create_argparser())
    factory.parse_args()
    print('Running %d combination(s)' % factory.max_combinations())
    parallel_run(factory, run_one_combination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
