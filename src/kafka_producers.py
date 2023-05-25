import json
import logging
import pickle

from confluent_kafka import Producer


def create_configuration_dict(**kwargs):
    return {k.replace('_', '.'): v for k, v in kwargs.items()}


class F1TopicProducer(Producer):
    def __init__(self, topic, dry_run=False, dry_run_verbose=False, value_serializer=pickle.dumps,
                 bootstrap_servers=('localhost:9092'), message_max_bytes=20 * 1024 * 1024, **kwargs):
        super(F1TopicProducer, self).__init__(
            create_configuration_dict(
                bootstrap_servers=bootstrap_servers, message_max_bytes=message_max_bytes, **kwargs
            )
        )
        self.topic = topic
        self.dry_run = dry_run
        self.dry_run_verbose = dry_run_verbose
        self.value_serializer = value_serializer

    def send_message(self, f1_topic):
        if self.dry_run:
            value = self.value_serializer(f1_topic)
            logging.info('Producer.produce(topic=%s key=%s, value_size=%d)' % (
                self.topic, f1_topic.topic_key(), len(value)
            ))
            if self.dry_run_verbose:
                logging.debug(json.dumps(f1_topic.__dict__, default=str, ensure_ascii=False, indent=4))
        else:
            self.produce(
                self.topic, key=f1_topic.topic_key(), value=self.value_serializer(f1_topic)
            )
            self.flush()
