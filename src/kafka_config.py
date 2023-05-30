import configparser
import logging
import sys

from collections import defaultdict
from f1_logging import init_logging


DEFAULTS_CLIENTS = 'clients'
DEFAULTS_TOPICS = 'topics'
PREFIX_CLIENT = 'client.'
PREFIX_DEFAULTS = 'defaults.'
PREFIX_TOPIC = 'topic.'
CLIENTS_ADMIN = PREFIX_CLIENT + 'admin'
CLIENTS_CONSUMER = PREFIX_CLIENT + 'consumer'
CLIENTS_PRODUCER = PREFIX_CLIENT + 'producer'


def parse_defaults(config):
    defaults = defaultdict(dict)
    for section in config.sections():
        if not section.startswith(PREFIX_DEFAULTS):
            continue
        default_subsection = '.'.join(section.split('.')[1:])
        for option in sorted(config[section]):
            defaults[default_subsection][option] = config[section][option]
            logging.debug('[default] %s:%s=%s' % (default_subsection, option, config[section][option]))
    return defaults


def parse_client_configurations(config, defaults):
    client_defaults = defaults.get(DEFAULTS_CLIENTS, {})
    for section in config.sections():
        if not section.startswith(PREFIX_CLIENT):
            continue
        logging.debug('Loading client configuration')
        for setting, value in client_defaults.items():
            if setting not in config[section]:
                config[section][setting] = str(value)
        for option in sorted(config[section]):
            logging.debug('[%s] %s=%s' % (section, option, config[section][option]))


def parse_topic_configurations(config, defaults):
    topic_defaults = defaults.get(DEFAULTS_TOPICS, {})
    for section in config.sections():
        if not section.startswith('topic.'):
            continue
        topic_name = section.split('.', maxsplit=1)[1]
        logging.debug('Loading configuration for topic "%s"' % topic_name)
        for setting, value in topic_defaults.items():
            if setting not in config[section]:
                config[section][setting] = str(value)
        for option in sorted(config[section]):
            logging.debug('[%s] %s=%s' % (section, option, config[section][option]))


def parse_configuration_file(filename):
    with open(filename, 'r') as infile:
        config_string = infile.read()
    return parse_configuration_string(config_string)


def parse_configuration_string(config_string):
    config = configparser.ConfigParser()
    config.read_string(config_string)
    defaults = parse_defaults(config)
    parse_client_configurations(config, defaults)
    parse_topic_configurations(config, defaults)
    return config


def get_configuration_dict(config, section):
    if section not in config:
        return {}
    return {k: config[section][k] for k in config[section]}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <config_filename>' % sys.argv[0])
        sys.exit(1)
    init_logging('config-test', loglevel=logging.DEBUG)
    _ = parse_configuration_file(sys.argv[1])
    sys.exit(0)
