import logging

import kafka_topic_names
import pickle

from confluent_kafka import Consumer, OFFSET_BEGINNING
from confluent_kafka.serialization import StringDeserializer


def create_configuration_dict(**kwargs):
    d = {k.replace('_', '.'): v for k, v in kwargs.items()}
    logging.debug(d)
    return d


class F1Message(object):
    def __init__(self, message, deserialized_value):
        self.topic = message.topic()
        self.partition = message.partition()
        self.offset = message.offset()
        self.value = deserialized_value


class F1TopicConsumer(Consumer):
    def __init__(self, topic, bootstrap_servers=('localhost:9092'), value_deserializer=pickle.loads, group_id='default',
                 read_from_beginning=False, max_partition_fetch_bytes=20 * 1024 * 1024, **kwargs):
        super(F1TopicConsumer, self).__init__(
            create_configuration_dict(
                bootstrap_servers=bootstrap_servers, group_id=group_id,
                max_partition_fetch_bytes=max_partition_fetch_bytes, **kwargs)
        )
        self.topic = topic
        self.read_from_beginning = read_from_beginning
        self.subscribe([topic], on_assign=self.on_assign, on_revoke=self.on_revoke)
        self.group_id = group_id
        self.key_serializer = StringDeserializer('utf_8')
        self.value_deserializer = value_deserializer
        self.current_partition_numbers = set()

    def consume_message(self, timeout=-1):
        messages = self.consume(num_messages=1, timeout=timeout)
        if not messages:
            return None
        assert len(messages) == 1, 'Requested 1 message but got %d' % len(messages)
        message = messages[0]
        if message.error():
            logging.error('Consumer error: {}'.format(message.error()))
            return None
        assert self.topic == message.topic(), 'Invalid message topic; expected "%s" but got "%s"' % (
            self.topic, message.topic()
        )
        return F1Message(message, self.value_deserializer(message.value()))

    def on_assign(self, consumer, partitions):
        logging.info('on_assign(%s, %s)' % (str(consumer), str(partitions)))
        assert self == consumer
        self.current_partition_numbers.update([p.partition for p in partitions])
        if self.read_from_beginning:
            for partition in partitions:
                partition.offset = OFFSET_BEGINNING
        self.assign(partitions)

    def on_revoke(self, consumer, partitions):
        logging.info('on_revoke(%s, %s)' % (str(consumer), str(partitions)))
        assert self == consumer
        for partition in partitions:
            assert partition.partition in self.current_partition_numbers
            self.current_partition_numbers.remove(partition.partition)


if __name__ == '__main__':
    topic_consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_SIM_RUNS, group_id='kafka-consumers-main-test',
                                     read_from_beginning=True)
    msg = topic_consumer.consume_message()
    while msg is not None:
        logging.debug(msg)
        msg = topic_consumer.consume_message()
