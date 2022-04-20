import argparse
import args
import calculator
import data_loader
import math
import sys

from parallel_worker import parallel_run


def create_argparser():
    parser = argparse.ArgumentParser(description='Formula 1 rating parameters', fromfile_prefix_chars='@')
    args.add_common_positional_args(parser)
    parser.add_argument('--print_future_simulations',
                        help='Print a file logging results of simulated future results.',
                        default=False, action='store_true')
    parser.add_argument('--print_ratings',
                        help='Print files logging ratings for teams and drivers.',
                        default=True, action='store_true')
    args.add_common_args(parser)
    return parser


def run_one_combination(parsed_args, task_queue):
    base_path = args.create_base_path(parsed_args)
    if parsed_args.print_args:
        args.print_args(parsed_args, base_path + '.args')
    precision = math.ceil(math.log10(parsed_args.run_max))
    print_str = '[ %%%dd / %%%dd ] %%s' % (precision, precision)
    print(print_str % (parsed_args.run_index, parsed_args.run_max, base_path))
    with data_loader.DataLoader(parsed_args, base_path) as loader:
        loader.load_events(parsed_args.events_tsv)
        loader.load_drivers(parsed_args.drivers_tsv)
        if not loader.load_teams(parsed_args.teams_tsv):
            print('ERROR during %s' % base_path)
            if task_queue is not None:
                task_queue.task_done()
            return True
        loader.load_results(parsed_args.results_tsv)
        rating_calculator = calculator.Calculator(parsed_args, base_path)
        rating_calculator.run_all_ratings(loader)
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
