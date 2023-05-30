import kafka_config
import kafka_topic_names
import logging
import pathlib
import sched
import sys

from collections import defaultdict
from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer
from time import monotonic, sleep, time


def validate_output_directory(directory_name):
    dir_obj = pathlib.Path(directory_name)
    if not dir_obj.exists():
        logging.error('Error: Output directory %s does not exist' % directory_name)
        return False
    if not dir_obj.is_dir():
        logging.error('Error: Output directory %s is not a directory' % directory_name)
        return False
    test_path = dir_obj / '.test'
    try:
        with open(test_path, 'a') as _:
            pass
        test_path.unlink()
        return True
    except IOError as error:
        logging.critical('Error: Output directory %s is not writable by this user: %s' % (directory_name, str(error)))
        return False


class SimulationFileWriter(object):
    def __init__(self, consumer, base_directory, close_on_timeout=120):
        self.consumer = consumer
        self.base_directory = base_directory
        self.close_on_timeout = close_on_timeout
        # [run_id] = file_handle
        self._file_handles = dict()
        # [run_id] = last_used
        self._file_timestamps = dict()
        # [run_id] = num_written
        self._sims_written = defaultdict(int)
        self._scheduler = sched.scheduler(monotonic, sleep)
        self._scheduler.enter(self.close_on_timeout / 2, 0, self.do_housekeeping)
        self._last_housekeeping = time()
        self._last_housekeeping_count = 0

    def process_messages(self):
        msg = self.consumer.consume_message(timeout=self.close_on_timeout / 4)
        while True:
            if msg is not None:
                self.process_one_message(msg)
            self._scheduler.run(blocking=False)
            msg = self.consumer.consume_message(timeout=self.close_on_timeout / 4)

    def process_one_message(self, message):
        topic = message.topic
        sim_output_block = message.value
        assert topic == kafka_topic_names.SANDBOX_SIM_OUTPUT, 'received message from wrong topic: %s' % topic
        logging.info('SimLogger.consume(topic="%s" key="%s" subkey="%s")' % (
            topic, sim_output_block.topic_key(), sim_output_block.topic_subkey()
        ))
        self.process_one_record(sim_output_block)

    def process_one_record(self, sim_output_block):
        run_id = sim_output_block.run_id
        output_file = self.fetch_file_handle(run_id)
        for sim in sim_output_block.simulations:
            print(sim.to_string(), file=output_file)
        self._sims_written[run_id] += len(sim_output_block.simulations)

    def fetch_file_handle(self, run_id):
        if run_id not in self._file_handles:
            logging.debug('open_file_for_run %d' % run_id)
            self._file_handles[run_id] = self.open_file_for_run(run_id)
        self._file_timestamps[run_id] = time()
        return self._file_handles[run_id]

    def open_file_for_run(self, run_id):
        filename = 'simulate-%s.simulations' % run_id
        output_path = pathlib.Path(self.base_directory) / filename
        return output_path.open(mode='a')

    def do_housekeeping(self):
        self._scheduler.enter(self.close_on_timeout / 2, 0, self.do_housekeeping)
        self.print_stats()
        self.clean_unused_handles()

    def print_stats(self):
        current_time = time()
        current_count = sum([n for n in self._sims_written.values()])
        duration = current_time - self._last_housekeeping
        recent_written = current_count - self._last_housekeeping_count
        logging.info('Wrote %d records in the last %.0f seconds [ %6.1f records/sec ]' % (
            recent_written, duration, recent_written / duration
        ))
        self._last_housekeeping = current_time
        self._last_housekeeping_count = current_count

    def clean_unused_handles(self):
        remove_older_than = time() - self.close_on_timeout
        unused_run_ids = [
            run_id for run_id, timestamp in self._file_timestamps.items() if timestamp < remove_older_than
        ]
        logging.debug('clean_unused_handles ~> %d of %d' % (len(unused_run_ids), len(self._file_timestamps)))
        for run_id in unused_run_ids:
            logging.debug('  close_unused(run_id=%d, sims_written=%d)' % (run_id, self._sims_written.get(run_id, 0)))
            self._file_handles[run_id].close()
            del self._file_handles[run_id]
            del self._file_timestamps[run_id]


def main(argv):
    if len(argv) != 3:
        print('Usage: %s <kafka_cfg> <output_directory>' % argv[0], file=sys.stderr)
        return 1
    output_directory = argv[2]
    init_logging('sim-logger')
    configuration = kafka_config.parse_configuration_file(argv[1])
    consumer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_CONSUMER)
    if not validate_output_directory(output_directory):
        return 1
    consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_SIM_OUTPUT, group_id='sim-logger', **consumer_config)
    file_writer = SimulationFileWriter(consumer, output_directory)
    file_writer.process_messages()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
