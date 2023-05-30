import json
import kafka_config
import logging
import sys

from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer


def main(argv):
    if len(argv) != 4:
        print('Usage: %s <config_txt> <topic_id> <run_group_id>' % argv[0])
        return 1
    topic_id = argv[2]
    group_id = argv[3]
    init_logging(group_id)
    configuration = kafka_config.parse_configuration_file(argv[1])
    consumer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_CONSUMER)
    consumer = F1TopicConsumer(topic_id, group_id=group_id, read_from_beginning=True, auto_offset_reset='smallest',
                               **consumer_config)
    msg = consumer.consume_message()
    while msg is not None:
        logging.info(json.dumps(msg.value.__dict__, default=str, ensure_ascii=False, indent=4))
        msg = consumer.consume_message()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
