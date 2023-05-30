import kafka_config
import logging
import sys

from confluent_kafka.admin import AdminClient, NewTopic
from f1_logging import init_logging


def topic_name_from_section(section_name):
    assert section_name.startswith(kafka_config.PREFIX_TOPIC)
    return section_name.split('.', maxsplit=1)[-1]


def create_one_topic_object(section_name, topic_config):
    topic_name = topic_name_from_section(section_name)
    num_partitions = int(topic_config.get('num_partitions'))
    replication_factor = int(topic_config.get('replication_factor'))
    other_configs = {k: v for k, v in topic_config.items() if k not in ['num_partitions', 'replication_factor']}
    return NewTopic(
        topic_name, num_partitions=num_partitions, replication_factor=replication_factor, config=other_configs
    )


def create_topics(config, admin_client):
    new_topics = [
        create_one_topic_object(section, config[section])
        for section in config.sections() if section.startswith(kafka_config.PREFIX_TOPIC)
    ]
    futures = admin_client.create_topics(new_topics)
    for topic, future in futures.items():
        try:
            future.result()  # The result itself is None
            logging.info("Topic {} created".format(topic))
        except Exception as e:
            logging.error("Failed to create topic {}: {}".format(topic, e))


def apply_config(config, admin_client):
    # TODO: First check for existing topics and sync properties as-needed.
    create_topics(config, admin_client)


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <config_file>' % argv[0], file=sys.stderr)
        return 1

    init_logging('kafka-admin-setup')
    configuration = kafka_config.parse_configuration_file(argv[1])
    admin_client = AdminClient(kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_ADMIN))
    apply_config(configuration, admin_client)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
