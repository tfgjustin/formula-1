import json
import kafka_config
import kafka_topic_names
import logging
import sys

from calculator import Calculator
from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer
from kafka_data_loader_cache import DataLoaderCache
from kafka_producers import F1TopicProducer
from kafka_topics import SimulationOutputBlock


def maybe_fetch_message(consumer, timeout, callback, verbose=False):
    record = consumer.consume_message(timeout=timeout)
    if record is None:
        return False
    topic = record.topic
    message = record.value
    logging.info(
        'Simulator.consume(topic="%s" key="%s" subkey="%s")' % (topic, message.topic_key(), message.topic_subkey())
    )
    if verbose:
        logging.debug(json.dumps(message.__dict__, default=str, ensure_ascii=False, indent=4))
    callback(record)
    return True


class Simulator(object):
    def __init__(self, ratings_consumer, sim_fuzz_consumer, sim_output_producer):
        self.data_loader_cache = DataLoaderCache(ratings_consumer)
        self.ratings_consumer = ratings_consumer
        self.sim_fuzz_consumer = sim_fuzz_consumer
        self.sim_output_producer = sim_output_producer
        # [run_id] = list()
        self.unassigned_blocks = dict()
        # Message fetch timeouts
        self._fetch_timeout_fuzzer = 4
        self._fetch_timeout_ratings = 0.02
        self._fetch_timeout_max = 4

    def process_messages(self):
        while True:
            # TODO: Figure out better loop termination criteria
            self.maybe_fetch_ratings()
            self.maybe_fetch_tag()

    def maybe_fetch_ratings(self, timeout=0):
        if maybe_fetch_message(self.ratings_consumer, timeout, self.process_ratings, verbose=False):
            # We actually did find ratings. Bring down the timeout for the fuzzer since we might need to grab another
            # rating next time around.
            self._fetch_timeout_fuzzer = self._fetch_timeout_fuzzer / 2
        else:
            # We did not find a rating, so bump up the timeout for the fuzzer
            self._fetch_timeout_fuzzer = min([self._fetch_timeout_max, self._fetch_timeout_fuzzer * 2])

    def maybe_fetch_tag(self):
        if maybe_fetch_message(self.sim_fuzz_consumer, self._fetch_timeout_fuzzer, self.process_sim_fuzz_message):
            # We did find fuzz, so wait less on the ratings
            self._fetch_timeout_ratings = self._fetch_timeout_ratings / 2
        else:
            # We did not find fuzz, so we can wait a bit more for the ratings
            self._fetch_timeout_ratings = min([self._fetch_timeout_max, self._fetch_timeout_ratings * 2])

    def process_ratings(self, message):
        sim_ratings = message.value
        self.data_loader_cache.add(sim_ratings.run_id, sim_ratings.data_loader, message.partition, message.offset)
        self.maybe_process_unassigned(sim_ratings.run_id)

    def maybe_process_unassigned(self, run_id):
        if run_id not in self.data_loader_cache:
            return
        if run_id not in self.unassigned_blocks:
            return
        for fuzz_request in self.unassigned_blocks[run_id]:
            self.process_sim_fuzz(fuzz_request)
        del self.unassigned_blocks[run_id]

    def process_sim_fuzz_message(self, message):
        self.process_sim_fuzz(message.value)

    def process_sim_fuzz(self, fuzz_block):
        if fuzz_block.run_id not in self.data_loader_cache:
            if fuzz_block.run_id not in self.unassigned_blocks:
                self.unassigned_blocks[fuzz_block.run_id] = list()
            self.unassigned_blocks[fuzz_block.run_id].append(fuzz_block)
            return
        loader = self.data_loader_cache.get(fuzz_block.run_id)
        # We have to do this since the num_iterations in the args is for the whole sim, not just this block
        self.sim_future_events(loader, fuzz_block)

    def sim_future_events(self, loader, fuzz_block):
        loader._args.num_iterations = fuzz_block.block_size
        c = Calculator(loader._args, None)
        output_buffer = list()
        c.simulate_future_events(loader, sim_log_idx_offset=fuzz_block.start_idx, fuzz=fuzz_block.fuzz,
                                 output_buffer=output_buffer)
        end_idx = fuzz_block.start_idx + fuzz_block.block_size - 1
        output_block = SimulationOutputBlock(run_id=fuzz_block.run_id, start_idx=fuzz_block.start_idx, end_idx=end_idx)
        output_block.extend(output_buffer)
        self.sim_output_producer.send_message(output_block)


def main(argv):
    if len(argv) != 3:
        print('Usage: %s <config_txt> <run_group_id>' % argv[0])
        return 1
    group_id = argv[2]
    init_logging(group_id)
    configuration = kafka_config.parse_configuration_file(argv[1])
    consumer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_CONSUMER)
    producer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_PRODUCER)
    ratings_consumer = F1TopicConsumer(
        kafka_topic_names.SANDBOX_SIM_RATINGS, group_id=group_id, read_from_beginning=True, **consumer_config
    )
    sim_fuzz_consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_SIM_FUZZ, group_id='simulators', **consumer_config)
    sim_output_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_OUTPUT, dry_run=False, dry_run_verbose=True,
                                          **producer_config)
    simulator = Simulator(ratings_consumer, sim_fuzz_consumer, sim_output_producer)
    simulator.process_messages()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
