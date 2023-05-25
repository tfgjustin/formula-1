import configparser
import json
import kafka_topic_names
import logging
import psycopg2
import sched
import sys
import tracemalloc

from collections import defaultdict
from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer
from time import monotonic, sleep, time
from kafka_topics import TagOutputAggregator


class RunResultArrays(object):
    def __init__(self, run_id=None, entity_id=None, event_id=None):
        self.run_id = run_id
        self.entity_id = entity_id
        self.event_id = event_id
        # [prop_id] = [list of sim IDs]
        self.tag_buckets = defaultdict(set)

    def add_tag_output_shard(self, tag_output_shard):
        # [prop_id] = [list of sim IDs]
        for prop_id, sim_ids in tag_output_shard.tags.items():
            self.tag_buckets[prop_id].update(sim_ids)

    def to_sql_insert_statements(self):
        if not self.tag_buckets:
            return None
        statements = [self.sql_insert_for_one_tag(tag) for tag in self.tag_buckets]
        logging.debug('Found a total of %d statements for [run_id=%s entity_id=%s event_id=%s]' % (
            len(statements), self.run_id, self.entity_id, self.event_id
        ))
        return statements

    def sql_insert_for_one_tag(self, tag):
        sim_ids = self.tag_buckets.get(tag)
        if sim_ids is None:
            return None
        insert_string = 'INSERT INTO run_results (run_id, event_id, entity_id, prop_id, sim_ids) VALUES '
        values = '(%s, \'%s\', \'%s\', \'%s\', ARRAY[%s])' % (
            self.run_id, self.event_id, self.entity_id, tag, ','.join([str(s) for s in sim_ids])
        )
        return insert_string + values


def maybe_fetch_message(consumer, timeout, callback, verbose=False):
    record = consumer.consume_message(timeout=timeout)
    if record is None:
        return
    logging.info('DbWriter.consume(topic="%s" key="%s" subkey="%s")' % (
        record.topic, record.value.topic_key(), record.value.topic_subkey()
    ))
    if verbose:
        logging.info(json.dumps(record.value.__dict__, default=str, ensure_ascii=False, indent=4))
    callback(record.value)


class DbWriter(object):
    _SECTION = 'postgresql'

    def __init__(self, run_consumer, tag_consumer):
        self.run_consumer = run_consumer
        self.tag_consumer = tag_consumer
        self.connection = None
        self.cursor = None
        self.config_parser = configparser.ConfigParser()
        # [run_id] = SimulationRun
        self.runs = dict()
        # [run_id][entity_id] = TagOutputAggregator
        self.aggregators = defaultdict(dict)
        # [run_id] = [list of EntityTagOutput topics not (yet) associated with a current run]
        self.unassigned_tags = dict()
        # Trace the program's memory usage
        self._trace_snapshot = None
        self._scheduler = sched.scheduler(monotonic, sleep)
        # Ignore event handler used for cancellation
        _ = self._scheduler.enter(0, 0, self.execute_scheduled_tasks)

    def open_connection(self, configfile):
        if self.connection is not None:
            return True
        if configfile is None:
            return False
        self.config_parser.read(configfile)
        if DbWriter._SECTION not in self.config_parser.sections():
            return False
        dbname = self.config_parser[DbWriter._SECTION]['dbname']
        user = self.config_parser[DbWriter._SECTION]['user']
        password = self.config_parser[DbWriter._SECTION]['password']
        host = self.config_parser[DbWriter._SECTION]['host']
        port = self.config_parser[DbWriter._SECTION]['port']
        try:
            self.connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            self.cursor = self.connection.cursor()
        except psycopg2.Error as e:
            logging.error(e)
            logging.error('Postgres: %s' % str(e.pgerror))
        return self.connection is not None

    def process_messages(self):
        while True:
            # TODO: Figure out better loop termination criteria
            self.maybe_fetch_run()
            self.maybe_fetch_tags()
            self._scheduler.run(blocking=False)

    def maybe_fetch_run(self, timeout=0.0):
        maybe_fetch_message(self.run_consumer, timeout, self.process_new_run, verbose=True)

    def maybe_fetch_tags(self, timeout=2):
        maybe_fetch_message(self.tag_consumer, timeout, self.process_tag_output_shard, verbose=False)

    def process_new_run(self, sim_run):
        if sim_run.run_id in self.runs:
            # TODO: Figure out how to merge in updates
            logging.warning('Duplicate run:', sim_run.run_id)
            return
        self.runs[sim_run.run_id] = sim_run
        if sim_run.run_id in self.unassigned_tags:
            for tag_output_shard in self.unassigned_tags[sim_run.run_id]:
                self.process_tag_output_shard(tag_output_shard)
            del self.unassigned_tags[sim_run.run_id]
        self.write_run(sim_run)

    def write_run(self, sim_run):
        if self.connection is None:
            return
        statement = 'INSERT INTO sim_runs (run_id, num_sims) VALUES (%d, %d)' % (sim_run.run_id, sim_run.total_num_sims)
        statement += ' ON CONFLICT ON CONSTRAINT sim_runs_pkey DO NOTHING'
        total_start = time()
        self.cursor.execute(statement)
        self.connection.commit()
        logging.debug('Total: %10.3fs' % (time() - total_start))

    def process_tag_output_shard(self, tag_output_shard):
        run_id = tag_output_shard.run_id
        sim_run = self.runs.get(run_id)
        if sim_run is None:
            self.process_tag_output_shard_unknown_run(run_id, tag_output_shard)
        else:
            self.process_tag_output_shard_with_run(sim_run, tag_output_shard)

    def process_tag_output_shard_unknown_run(self, run_id, tag_output_shard):
        if run_id not in self.unassigned_tags:
            self.unassigned_tags[run_id] = list()
        self.unassigned_tags[run_id].append(tag_output_shard)

    def process_tag_output_shard_with_run(self, sim_run, tag_output_shard):
        run_id = sim_run.run_id
        entity_id = tag_output_shard.entity_id
        event_id = tag_output_shard.event_id
        if entity_id not in self.aggregators[run_id]:
            self.aggregators[run_id][entity_id] = dict()
        if event_id not in self.aggregators[run_id][entity_id]:
            self.aggregators[run_id][entity_id][event_id] = TagOutputAggregator(
                run_id=run_id, entity_id=entity_id, event_id=event_id, total_num_sims=sim_run.total_num_sims
            )
        tag_output_aggregator = self.aggregators[run_id][entity_id][event_id]
        tag_output_aggregator.add_tag_output_shard(tag_output_shard)
        self.maybe_write_entity_and_cleanup(tag_output_aggregator)

    def maybe_write_entity_and_cleanup(self, tag_output_aggregator):
        if not tag_output_aggregator.is_complete:
            return
        # Write information about this entity to the database
        run_id = tag_output_aggregator.run_id
        entity_id = tag_output_aggregator.entity_id
        event_id = tag_output_aggregator.event_id
        factory = RunResultArrays(run_id=run_id, entity_id=entity_id, event_id=event_id)
        for tag_output_shard in tag_output_aggregator.tag_shards():
            factory.add_tag_output_shard(tag_output_shard)
        self.write_result(factory)
        self.recursive_cleanup(run_id, entity_id, event_id)

    def write_result(self, run_results):
        if self.connection is None:
            return
        total_start = time()
        n = 0
        for statement in run_results.to_sql_insert_statements():
            if statement is None or not statement:
                continue
            start_time = time()
            logging.debug(statement)
            self.cursor.execute(statement)
            self.connection.commit()
            end_time = time()
            logging.debug('Statement %05d: %10.3fs Total: %10.3fs' % (n, end_time - start_time, end_time - total_start))
            n += 1
        if n:
            logging.info('Total SQL execution time: %10.3fs' % (time() - total_start))

    def recursive_cleanup(self, run_id, entity_id, event_id):
        del self.aggregators[run_id][entity_id][event_id]
        if not self.aggregators[run_id][entity_id]:
            del self.aggregators[run_id][entity_id]
        if not self.aggregators[run_id]:
            del self.aggregators[run_id]

    def close_connection(self):
        if self.connection is None:
            return
        self.cursor.close()
        self.connection.close()

    def execute_scheduled_tasks(self):
        # First we schedule the next instance
        self._scheduler.enter(180, 0, self.execute_scheduled_tasks)
        self.print_stats()

    def print_stats(self):
        # TODO: Skip this whole thing if the logging level is not at debug.
        # This will save a ton of CPU time spent trying to grab malloc snapshots.
        self.print_dict_stats()
        self.print_memory_stats()

    def print_dict_stats(self):
        # [run_id] = SimulationRun
        logging.debug('Dict %s print_dict_stats.runs %d' % (self.run_consumer.group_id, len(self.runs)))
        # [run_id][entity_id] = TagOutputAggregator
        logging.debug('Dict %s print_dict_stats.aggregators %d %d' % (
            self.run_consumer.group_id, len(self.aggregators), sum(
                [len(tag_output_aggregators) for tag_output_aggregators in self.aggregators.values()]
            )
        ))
        # [run_id] = [list of EntityTagOutput topics not (yet) associated with a current run]
        logging.debug('Dict %s print_dict_stats.unassigned_tags %d %d' % (
            self.run_consumer.group_id, len(self.unassigned_tags),
            sum([len(v) for v in self.unassigned_tags.values()])
        ))

    def print_memory_stats(self):
        snapshot = tracemalloc.take_snapshot()
        self.print_memory_stats_top(snapshot)
        self.print_memory_stats_traceback(snapshot)
        self.print_memory_stats_diff(snapshot)
        self._trace_snapshot = snapshot

    def print_memory_stats_top(self, current_snapshot):
        top_stats = current_snapshot.statistics('lineno')
        for stat in top_stats[:2]:
            logging.debug('Tops %s %s' % (self.run_consumer.group_id, str(stat)))

    def print_memory_stats_traceback(self, current_snapshot):
        traces = current_snapshot.statistics('traceback')
        for stat in sorted(traces, key=lambda s: s.size, reverse=True)[:2]:
            logging.debug('Trace %s blocks=%d size_MiB=%.3f' % (
                self.run_consumer.group_id, stat.count, (stat.size / (1024 * 1024))
            ))
            for line in stat.traceback.format():
                logging.debug('Trace %s %s' % (self.run_consumer.group_id, str(line)))

    def print_memory_stats_diff(self, current_snapshot):
        if self._trace_snapshot is None:
            return
        diff_stats = current_snapshot.compare_to(self._trace_snapshot, 'lineno')
        for stat in diff_stats[:2]:
            logging.debug('Diff %s %s' % (self.run_consumer.group_id, str(stat)))


def main(argv):
    if len(argv) != 3:
        print('Usage: %s <db_cfg> <run_group_id>' % argv[0])
        return 1
    init_logging(argv[2])
    tracemalloc.start(3)
    run_consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_SIM_RUNS, group_id=argv[2], read_from_beginning=True)
    tag_consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_TAG_OUTPUT, group_id='db-writers')
    db_writer = DbWriter(run_consumer, tag_consumer)
    if not db_writer.open_connection(argv[1]):
        return 1
    db_writer.process_messages()
    db_writer.close_connection()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
