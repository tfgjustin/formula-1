import kafka_topic_names
import logging
import sched

from confluent_kafka import TopicPartition
from time import monotonic, sleep, time


class DataLoaderCache(object):
    def __init__(self, data_loader_consumer, num_cached=4, max_age_seconds=120):
        # The consumer fetching the data loaders
        self._data_loader_consumer = data_loader_consumer
        self._group_id = data_loader_consumer.group_id
        self._num_cached = num_cached
        # We also evict items from the cache which haven't been accessed in a certain amount of time
        self._max_age_seconds = max_age_seconds
        # Map of run ID to data loaders
        self._data_loaders = dict()
        # Map of run ID to TopicPartition objects
        self._data_loader_offsets = dict()
        # Map of run ID to when they were fetched
        self._last_fetched = dict()
        self._scheduler = sched.scheduler(monotonic, sleep)
        self._scheduler.enter(self._max_age_seconds / 2, 0, self.evict_unused)

    def __contains__(self, run_id):
        return run_id in self._data_loader_offsets

    def add(self, run_id, data_loader, partition_num, partition_offset, topic=kafka_topic_names.SANDBOX_SIM_RATINGS):
        self._scheduler.run(blocking=False)
        logging.info('%s LoaderCache.add(run_id=%d partition=%d offset=%d)' % (
            self._group_id, run_id, partition_num, partition_offset
        ))
        if run_id in self._data_loader_offsets:
            logging.info('%s LoaderCache.add(run_id=%d) ~> duplicate' % (self._group_id, run_id))
            return
        self._data_loaders[run_id] = data_loader
        self._last_fetched[run_id] = time()
        self._data_loader_offsets[run_id] = TopicPartition(topic, partition_num, offset=partition_offset)
        # See if we need to evict anything
        self.update_cache()

    def get(self, run_id):
        self._scheduler.run(blocking=False)
        if run_id not in self._data_loader_offsets:
            # We've never seen this run_id before
            logging.warning('%s LoaderCache.get(run_id=%d) ~> (unknown)' % (self._group_id, run_id))
            return None
        if run_id in self._data_loaders:
            self._last_fetched[run_id] = time()
            logging.info('%s LoaderCache.get(run_id=%d) ~> (latest=%.2f)' % (
                self._group_id, run_id, self._last_fetched[run_id]
            ))
            return self._data_loaders[run_id]
        else:
            logging.info('%s LoaderCache.get(run_id=%d) ~> (page-in)' % (self._group_id, run_id))
            target_loader = self.fetch_from_topic(run_id)
            # We probably overfilled the cache, so possibly evict one cached object.
            self.update_cache()
            return target_loader

    def fetch_from_topic(self, run_id):
        # If we're here, then we've seen the data loader before but it wasn't in the cache. Try and fetch it.
        assert run_id in self._data_loader_offsets
        # Grab the current location for all the partitions to which we're connected.
        logging.debug(self._group_id,  'CacheLocation.current_partition_numbers ~>',
                      self._data_loader_consumer.current_partition_numbers)
        xy_partitions = [
            TopicPartition(self._data_loader_consumer.topic, partition_num)
            for partition_num in self._data_loader_consumer.current_partition_numbers
        ]
        logging.info(self._group_id, ' CacheLocation.fetch_from_topic( run_id =', run_id, ')', xy_partitions)
        current_partition_locations = {
            pos.partition: pos for pos in self._data_loader_consumer.committed(xy_partitions)
        }
        # Get the target partition location from which we'll fetch the DataLoader
        target_partition_location = self._data_loader_offsets[run_id]
        logging.info(
            '%s  CacheLocation(run_id=%d) ~> seek_do(partition=%d, offset=%d) [current (partition=%d offset=%d)]' % (
                self._group_id, run_id, target_partition_location.partition, target_partition_location.offset,
                target_partition_location.partition,
                current_partition_locations[target_partition_location.partition].offset
            )
        )
        # Assert that the target partition is actually in the set of partitions to which we're currently connected
        assert target_partition_location.partition in current_partition_locations
        # Seek to the correct offset and then load the data
        self._data_loader_consumer.seek(target_partition_location)
        record = self._data_loader_consumer.consume_message()
        assert record is not None
        assert record.value.run_id == run_id
        target_data_loader = record.value.data_loader
        logging.info(self._group_id, ' CacheFetch(run_id=%d) ~>' % run_id, type(target_data_loader))
        # Store this in our local cache
        self._data_loaders[run_id] = target_data_loader
        self._last_fetched[run_id] = time()
        logging.info(
            '%s  CacheFetch(run_id=%d) ~> success(%.2f)' % (self._group_id, run_id, self._last_fetched[run_id])
        )
        # Seek back to the original location
        self._data_loader_consumer.seek(current_partition_locations[target_partition_location.partition])
        logging.info('%s  CacheLocation(run_id=%d) ~> seek_undo(partition=%d offset=%d)' % (
            self._group_id, run_id,  current_partition_locations[target_partition_location.partition].partition,
            current_partition_locations[target_partition_location.partition].offset,
        ))
        return target_data_loader

    def update_cache(self):
        logging.info('%s LoaderCache.update_cache(size=%d max_size=%d)' % (
            self._group_id, len(self._data_loaders), self._num_cached
        ))
        if len(self._data_loaders) <= self._num_cached:
            logging.info('%s  (no-op)' % self._group_id)
            return
        # We need to evict something. Find the oldest one we fetched.
        while len(self._data_loaders) > self._num_cached:
            self.evict_oldest()
        logging.info('%s  LoaderCache.update_cache(size=%d max_size=%d)' % (
            self._group_id, len(self._data_loaders), self._num_cached
        ))

    def evict_oldest(self):
        logging.info('%s  (evict-oldest)' % self._group_id)
        # Sort the last fetched by the timestamp, then
        # [0] gets the oldest item
        # [0] gets the run ID of the oldest
        oldest_data_loader_run_id = sorted(self._last_fetched.items(), key=lambda ts: ts[1])[0][0]
        # We remove the data loader from memory and the last_fetched, but not the offsets. We'll need the offsets to
        # pull the data back into the cache later (maybe).
        self.evict_for_run_id(oldest_data_loader_run_id)

    def evict_unused(self):
        logging.info('%s LoaderCache.evict_unused(size=%d max_age=%d)' % (
            self._group_id, len(self._data_loaders), self._max_age_seconds
        ))
        self._scheduler.enter(self._max_age_seconds / 2, 0, self.evict_unused)
        # If a data loader has not been accessed since this time, evict it
        minimum_access_time = time() - self._max_age_seconds
        unused_run_ids = [
            run_id for run_id, last_access_time in self._last_fetched.items() if last_access_time < minimum_access_time
        ]
        for run_id in unused_run_ids:
            self.evict_for_run_id(run_id)

    def evict_for_run_id(self, run_id):
        logging.info('%s  LoaderCache.evict_for_run_id(run_id=%d)' % (self._group_id, run_id))
        assert run_id in self._data_loaders
        assert run_id in self._last_fetched
        del self._data_loaders[run_id]
        del self._last_fetched[run_id]
