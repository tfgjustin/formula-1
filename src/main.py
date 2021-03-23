import calculator
import data_loader
import math
import multiprocessing
import sys

from args import ArgFactory


def create_path(args):
    path = args.logfile
    if args.logfile_uses_parameters:
        path = '-'.join([
            args.logfile, str(args.driver_elo_initial), str(args.driver_elo_regress), str(args.driver_kfactor_regress),
            str(args.driver_reliability_decay), str(args.elo_compare_window), str(args.elo_exponent_denominator_race),
            str(args.elo_exponent_denominator_qualifying), str(args.qualifying_kfactor_multiplier),
            args.position_base_spec, str(args.position_base_factor), str(args.team_elo_initial),
            str(args.team_elo_regress), str(args.team_kfactor_regress), str(args.team_reliability_decay),
            args.team_share_spec
        ])
    return path


def create_outfile(args):
    path = create_path(args)
    return open(path, 'w')


class Worker(multiprocessing.Process):
    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self._task_queue = task_queue

    def run(self):
        while True:
            args = self._task_queue.get()
            if args is None:
                self._task_queue.task_done()
                break
            task_done = False
            try:
                base_path = create_path(args)
                precision = math.ceil(math.log10(args.run_max))
                print_str = '[ %%%dd / %%%dd ] %%s' % (precision, precision)
                print(print_str % (args.run_index, args.run_max, base_path))
                with data_loader.DataLoader(args, base_path) as loader:
                    loader.load_events(args.events_tsv)
                    loader.load_drivers(args.drivers_tsv)
                    if not loader.load_teams(args.teams_tsv):
                        print('ERROR %s' % base_path)
                        self._task_queue.task_done()
                        task_done = True
                        break
                    loader.load_results(args.results_tsv)
                    rating_calculator = calculator.Calculator(args, base_path)
                    rating_calculator.run_all_ratings(loader)
                self._task_queue.task_done()
                task_done = True
            finally:
                if not task_done:
                    self._task_queue.task_done()


def main():
    _NUM_THREADS = 12
    factory = ArgFactory()
    factory.parse_args()
    print('Running %d combinations' % factory.max_combinations())
    tasks = multiprocessing.JoinableQueue()
    workers = [Worker(tasks) for _ in range(_NUM_THREADS)]
    for w in workers:
        w.start()
    current_args = factory.next_config()
    while current_args is not None:
        tasks.put(current_args)
        current_args = factory.next_config()
    for _ in range(_NUM_THREADS):
        tasks.put(None)
    tasks.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
