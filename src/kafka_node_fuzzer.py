import json
import kafka_topic_names
import logging
import fuzz
import sys

from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer
from kafka_data_loader_cache import DataLoaderCache
from kafka_producers import F1TopicProducer
from kafka_topics import SimulationFuzzBlock


def maybe_fetch_message(consumer, timeout, callback, verbose=False):
    record = consumer.consume_message(timeout=timeout)
    if record is None:
        return False
    topic = record.topic
    message = record.value
    logging.info('Fuzzer.consume(topic="%s" key="%s")' % (topic, message.topic_key()))
    if verbose:
        logging.debug(json.dumps(message.__dict__, default=str, ensure_ascii=False, indent=4))
    callback(record)
    return True


class Fuzzer(object):
    def __init__(self, ratings_consumer, fuzz_request_consumer, fuzz_producer):
        self.data_loader_cache = DataLoaderCache(ratings_consumer)
        self.ratings_consumer = ratings_consumer
        self.fuzz_request_consumer = fuzz_request_consumer
        self.fuzz_producer = fuzz_producer
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

    def maybe_fetch_ratings(self):
        if maybe_fetch_message(self.ratings_consumer, self._fetch_timeout_ratings, self.process_ratings, verbose=False):
            # We actually did find ratings. Bring down the timeout for the fuzzer since we might need to grab another
            # rating next time around.
            self._fetch_timeout_fuzzer = self._fetch_timeout_fuzzer / 2
        else:
            # We did not find a rating, so bump up the timeout for the fuzzer
            self._fetch_timeout_fuzzer = min([self._fetch_timeout_max, self._fetch_timeout_fuzzer * 2])

    def maybe_fetch_tag(self):
        if maybe_fetch_message(self.fuzz_request_consumer, self._fetch_timeout_fuzzer,
                               self.process_fuzz_request_message, verbose=False):
            # We did find fuzz, so wait less on the ratings
            self._fetch_timeout_ratings = self._fetch_timeout_ratings / 2
        else:
            # We did not find fuzz, so we can wait a bit more for the ratings
            self._fetch_timeout_ratings = min([self._fetch_timeout_max, self._fetch_timeout_ratings * 2])

    def process_ratings(self, message):
        sim_ratings = message.value
        self.data_loader_cache.add(sim_ratings.run_id, sim_ratings.data_loader, message.partition, message.offset)
        self.maybe_process_unassigned(sim_ratings.run_id)

    def process_fuzz_request_message(self, message):
        self.process_fuzz_request(message.value)

    def process_fuzz_request(self, fuzz_request):
        if fuzz_request.run_id not in self.data_loader_cache:
            if fuzz_request.run_id not in self.unassigned_blocks:
                self.unassigned_blocks[fuzz_request.run_id] = list()
            self.unassigned_blocks[fuzz_request.run_id].append(fuzz_request)
            return
        loader = self.data_loader_cache.get(fuzz_request.run_id)
        # We have to do this since the num_iterations in the args is for the whole sim, not just this block
        self.fuzz_future_events(loader, fuzz_request)

    def maybe_process_unassigned(self, run_id):
        if run_id not in self.data_loader_cache:
            return
        if run_id not in self.unassigned_blocks:
            return
        for fuzz_request in self.unassigned_blocks[run_id]:
            self.process_fuzz_request(fuzz_request)
        del self.unassigned_blocks[run_id]

    def fuzz_future_events(self, loader, fuzz_request):
        loader._args.num_iterations = fuzz_request.block_size
        fuzzer = fuzz.Fuzzer(loader._args, None)
        future_seasons = loader.future_seasons()
        for year in sorted(future_seasons.keys()):
            events = future_seasons[year].events()
            drivers = loader.future_drivers()
            teams = loader.future_teams()
            fuzzer.generate_all_fuzz(year, events, drivers, teams)
        self.fuzz_producer.send_message(
            SimulationFuzzBlock(run_id=fuzz_request.run_id, start_idx=fuzz_request.start_idx,
                                block_size=fuzz_request.block_size, fuzz=fuzzer.all_fuzz())
        )


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <run_group_id>' % argv[0])
        return 1
    init_logging(argv[1])
    ratings_consumer = F1TopicConsumer(
        kafka_topic_names.SANDBOX_SIM_RATINGS, group_id=argv[1], read_from_beginning=True
    )
    fuzz_request_consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_FUZZ_REQUEST, group_id='fuzzers')
    fuzz_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_FUZZ, dry_run=False, dry_run_verbose=False)
    fuzzer = Fuzzer(ratings_consumer, fuzz_request_consumer, fuzz_producer)
    fuzzer.process_messages()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
