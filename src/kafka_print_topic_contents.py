import json
import logging
import sys

from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer


def main(argv):
    if len(argv) != 3:
        print('Usage: %s <topic_id> <run_group_id>' % argv[0])
        return 1
    init_logging(argv[2])
    consumer = F1TopicConsumer(argv[1], group_id=argv[2], read_from_beginning=True, auto_offset_reset='smallest')
    msg = consumer.consume_message()
    while msg is not None:
        logging.info(json.dumps(msg.value.__dict__, default=str, ensure_ascii=False, indent=4))
        msg = consumer.consume_message()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
