import argparse
import args
import data_loader
import kafka_config
import kafka_topic_names
import logging
import sys

from f1_logging import init_logging
from kafka_producers import F1TopicProducer
from kafka_topics import FuzzRequestBlock, SimulationInput, SimulationRun
from time import time_ns


def create_argparser():
    parser = argparse.ArgumentParser(description='Formula 1 simulator parameters', fromfile_prefix_chars='@')
    parser.add_argument('kafka_config_txt',
                        help='Configuration file containing Kafka-specific configuration data.',
                        type=argparse.FileType('r'), default='')
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
    parser.add_argument('--fuzz_block_size', help='Number of simulations in one fuzz block.',
                        type=int, default=20)
    args.add_common_args(parser)
    return parser


def create_run_id():
    return time_ns()


def validate_fuzz_block_size(parsed_args):
    if parsed_args.num_iterations % parsed_args.fuzz_block_size:
        logging.error('Invalid fuzz block size: %d simulations does not divide evenly into blocks of size %d' % (
            parsed_args.num_iterations, parsed_args.fuzz_block_size
        ))
        return False
    return True


def generate_fuzz_requests(parsed_args, run_id, sim_fuzz_producer):
    logging.debug('Generating fuzz')
    for start_idx in range(0, parsed_args.num_iterations, parsed_args.fuzz_block_size):
        fuzz_block = FuzzRequestBlock(run_id=run_id, start_idx=start_idx, block_size=parsed_args.fuzz_block_size)
        logging.debug('FuzzRequestBlock(run_id=%d;start_idx=%d;block_size=%d)' % (
            fuzz_block.run_id, fuzz_block.start_idx, fuzz_block.block_size
        ))
        sim_fuzz_producer.send_message(fuzz_block)


def publish_one_sim(parsed_args, sim_run_producer, sim_rating_producer, sim_fuzz_producer):
    if not validate_fuzz_block_size(parsed_args):
        return
    with data_loader.DataLoader(parsed_args, None) as loader:
        # Load the events
        loader.load_events(parsed_args.events_tsv)
        # Load the driver database
        loader.load_drivers(parsed_args.drivers_tsv)
        # Team history
        if not loader.load_teams(parsed_args.teams_tsv):
            return
        # Historical results
        loader.load_results(parsed_args.results_tsv)
        if not loader.load_grid_penalties(parsed_args.grid_penalties_tsv):
            return
        # Load historical driver data
        if not loader.update_drivers_from_ratings(parsed_args.driver_ratings_tsv):
            return
        # Load historical team data
        if not loader.load_teams_from_ratings(parsed_args.team_ratings_tsv):
            return
        # Load future simulation events and determine future lineups (file or last event)
        loader.load_future_simulation_data()
        run_id = create_run_id()
        logging.debug('run_id=%d' % run_id)
        sim_run = SimulationRun(run_id=run_id, total_num_sims=parsed_args.num_iterations)
        logging.debug('SimulationRun(run_id=%d;total_num_sims=%d)' % (sim_run.run_id, sim_run.total_num_sims))
        sim_run_producer.send_message(sim_run)
        sim_input = SimulationInput(run_id=run_id, data_loader=loader)
        logging.debug('SimulationInput(run_id=%d)' % run_id)
        sim_rating_producer.send_message(sim_input)
        logging.debug('SimulationInput: sent')
        generate_fuzz_requests(parsed_args, run_id, sim_fuzz_producer)
        logging.info('Initialized run_id=%d' % run_id)


def publish_sim_messages(arg_factory, current_args, sim_run_producer, sim_rating_producer, sim_fuzz_producer):
    while current_args is not None:
        publish_one_sim(current_args, sim_run_producer, sim_rating_producer, sim_fuzz_producer)
        current_args = arg_factory.next_config()


def main():
    init_logging('simulation-driver', loglevel=logging.DEBUG)
    factory = args.ArgFactory(create_argparser())
    factory.parse_args()
    logging.info('Running %d combination(s)' % factory.max_combinations())
    # We get the first args because it will have the Kafka configuration file.
    first_args = factory.next_config()
    configuration = kafka_config.parse_configuration_string(first_args.kafka_config_txt)
    producer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_PRODUCER)
    sim_run_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_RUNS, dry_run=False, dry_run_verbose=False,
                                       **producer_config)
    sim_rating_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_RATINGS, dry_run=False, dry_run_verbose=False,
                                          compression_type='gzip', **producer_config)
    sim_fuzz_producer = F1TopicProducer(kafka_topic_names.SANDBOX_FUZZ_REQUEST, dry_run=False, dry_run_verbose=False,
                                        **producer_config)
    publish_sim_messages(factory, first_args, sim_run_producer, sim_rating_producer, sim_fuzz_producer)
    return 0


if __name__ == '__main__':
    sys.exit(main())
