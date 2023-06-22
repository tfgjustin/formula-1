import multiprocessing

_NUM_THREADS = 10


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, run_one):
        multiprocessing.Process.__init__(self)
        self._task_queue = task_queue
        self._run_one = run_one

    def run(self):
        while True:
            args = self._task_queue.get()
            if args is None:
                self._task_queue.task_done()
                break
            task_done = False
            try:
                task_done = self._run_one(args, self._task_queue)
            finally:
                if not task_done:
                    self._task_queue.task_done()


def run_multiprocessor(factory, run_one_function):
    task_queue = multiprocessing.JoinableQueue()
    workers = [Worker(task_queue, run_one_function) for _ in range(_NUM_THREADS)]
    for w in workers:
        w.start()
    current_args = factory.next_config()
    while current_args is not None:
        task_queue.put(current_args)
        current_args = factory.next_config()
    for _ in range(_NUM_THREADS):
        task_queue.put(None)
    task_queue.join()


def run_in_current_process(factory, run_one_function):
    args = factory.next_config()
    if args is None:
        return
    run_one_function(args, None)


def parallel_run(factory, run_one_function):
    if factory.max_combinations() > 1:
        run_multiprocessor(factory, run_one_function)
    else:
        run_in_current_process(factory, run_one_function)
