import configparser

from confluent_kafka.admin import (AdminClient, NewTopic, NewPartitions)
import sys
import logging

logging.basicConfig()


_ADMIN_CLIENT_NAME = 'admin.client'
_TOPIC_PREFIX = 'topic.'


def topic_name_from_section(section_name):
    assert section_name.startswith(_TOPIC_PREFIX)
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
        for section in config.sections() if section.startswith(_TOPIC_PREFIX)
    ]
    futures = admin_client.create_topics(new_topics)
    for topic, future in futures.items():
        try:
            future.result()  # The result itself is None
            logging.info("Topic {} created".format(topic))
        except Exception as e:
            logging.error("Failed to create topic {}: {}".format(topic, e))


def parse_admin_client_configuration(config):
    _default_settings = {
        'bootstrap.servers': 'localhost:9092'
    }
    for section in config.sections():
        if section != _ADMIN_CLIENT_NAME:
            continue
        logging.debug('Loading admin client configuration')
        for setting, value in _default_settings.items():
            if setting not in config[section]:
                config[section][setting] = str(value)
        for option in sorted(config[section]):
            logging.debug('%s=%s' % (option, config[section][option]))


def parse_topic_configurations(config):
    _default_settings = {
        'num_partitions': 1,
        'replication_factor': 1,
        'retention.ms': 86400000,
    }
    for section in config.sections():
        if not section.startswith('topic.'):
            continue
        topic_name = section.split('.', maxsplit=1)[1]
        logging.debug('Loading configuration for topic "%s"' % topic_name)
        for setting, value in _default_settings.items():
            if setting not in config[section]:
                config[section][setting] = str(value)
        for option in sorted(config[section]):
            logging.debug('%s=%s' % (option, config[section][option]))


def parse_configuration(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    parse_admin_client_configuration(config)
    parse_topic_configurations(config)
    return config


def apply_config(config, admin_client):
    # TODO: First check for existing topics and sync properties as-needed.
    create_topics(config, admin_client)


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <config_file>' % argv[0], file=sys.stderr)
        return 1

    configuration = parse_configuration(argv[1])
    admin_config = {k: configuration[_ADMIN_CLIENT_NAME][k] for k in configuration[_ADMIN_CLIENT_NAME]}
    admin_client = AdminClient(admin_config)
    apply_config(configuration, admin_client)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
