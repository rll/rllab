import os
import sys
sys.path.append('.')
import threading
import time
import warnings
import multiprocessing
import importlib

from rllab import config
from rllab.misc.instrument import run_experiment_lite

import polling
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials

class S3SyncThread(threading.Thread):
    '''
    Thread to periodically sync results from S3 in the background.
    
    Uses same dirs as ./scripts/sync_s3.py.
    '''
    def __init__(self, sync_interval=60):
        super(S3SyncThread, self).__init__()
        self.sync_interval = sync_interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()
    
    def run(self):
        remote_dir = config.AWS_S3_PATH
        local_dir = os.path.join(config.LOG_DIR, "s3")
        command = ("""
            aws s3 sync {remote_dir} {local_dir} --exclude '*stdout.log' --exclude '*stdouterr.log' --content-type "UTF-8"
        """.format(local_dir=local_dir, remote_dir=remote_dir))
        while True:
            fail = os.system(command)
            if fail:
                warnings.warn("Problem running the s3 sync command. You might want to run ./scripts/sync_s3.py manually in a shell to inspect.")
            if self.stopped():
                break
            time.sleep(self.sync_interval)

def _launch_workers(exp_key, n_workers, host, port, result_db_name):
    jobs = []
    for i in range(n_workers):
        p = multiprocessing.Process(target=_launch_worker, args=(exp_key,i,host, port, result_db_name))
        jobs.append(p)
        p.start()
        time.sleep(1)
    return jobs
          
def _launch_worker(exp_key, worker_id, host, port, result_db_name):
    command = "hyperopt-mongo-worker --mongo={h}:{p}/{db} --poll-interval=10 --exp-key={key} > hyperopt_worker{id}.log 2>&1"
    command = command.format(h=host, p=port, db=result_db_name, key=exp_key, id=worker_id)
    fail = os.system(command)
    if fail:
        raise RuntimeError("Problem starting hyperopt-mongo-worker.")
    
def _wait_result(exp_prefix, exp_name, timeout):
    """
    Poll for the sync of params.pkl (currently hardcoded) from S3, indicating that the task is done.
    
    :param exp_prefix: str, experiment name prefix (dir where results are expected to be stored)
    :param exp_name: str, experiment name. Name of dir below exp_prefix where result files of individual run are
        expected to be stored
    :param timeout: int, polling timeout in seconds
    :return bool. False if the polling times out. True if successful.
    """
    result_path = os.path.join(config.LOG_DIR, "s3", exp_prefix, exp_name, 'params.pkl')
    print("Polling for results in",result_path) 
    try:
        file_handle = polling.poll(
            lambda: open(result_path),
            ignore_exceptions=(IOError,),
            timeout=timeout,
            step=60)
        file_handle.close()
    except polling.TimeoutException:
        return False
    return True

def _launch_ec2(func, exp_prefix, exp_name, params, run_experiment_kwargs):
    print("Launching task", exp_name)
    kwargs = dict(
        n_parallel=1,
        snapshot_mode="last",
        seed=params.get("seed",None),
        mode="ec2"
        )
    kwargs.update(run_experiment_kwargs)
    kwargs.update(dict(
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        variant=params,
        confirm_remote=False))
    
    run_experiment_lite(func,**kwargs)

def _get_stubs(params):
    module_str = params.pop('task_module')
    func_str = params.pop('task_function')
    eval_module_str = params.pop('eval_module')
    eval_func_str = params.pop('eval_function')
    
    module = importlib.import_module(module_str)
    func = getattr(module, func_str)
    eval_module = importlib.import_module(eval_module_str)
    eval_func = getattr(eval_module, eval_func_str)
    
    return func, eval_func
    
task_id = 1
def objective_fun(params):
    global task_id    
    exp_prefix = params.pop("exp_prefix")
    exp_name = "{exp}_{pid}_{id}".format(exp=exp_prefix, pid=os.getpid(), id=task_id)
    max_retries = params.pop('max_retries', 0) + 1
    result_timeout = params.pop('result_timeout')
    run_experiment_kwargs = params.pop('run_experiment_kwargs', {})
        
    func, eval_func = _get_stubs(params)

    result_success = False
    while max_retries > 0:
        _launch_ec2(func, exp_prefix, exp_name, params, run_experiment_kwargs)
        task_id += 1; max_retries -= 1
        if _wait_result(exp_prefix, exp_name, result_timeout):
            result_success = True
            break
        elif max_retries > 0:
            print("Timed out waiting for results. Retrying...")
    
    if not result_success:
        print("Reached max retries, no results. Giving up.")
        return {'status':STATUS_FAIL}
    
    print("Results in! Processing.")
    result_dict = eval_func(exp_prefix, exp_name)
    result_dict['status'] = STATUS_OK
    result_dict['params'] = params
    return result_dict


def launch_hyperopt_search(
        task_method,
        eval_method,
        param_space,
        hyperopt_experiment_key,
        hyperopt_db_host="localhost",
        hyperopt_db_port=1234,
        hyperopt_db_name="rllab",
        n_hyperopt_workers=1,
        hyperopt_max_evals=100,
        result_timeout=1200,
        max_retries=0,
        run_experiment_kwargs=None):
    """
    Launch a hyperopt search using EC2.
    
    This uses the hyperopt parallel processing functionality based on MongoDB. The MongoDB server at the specified host
    and port is assumed to be already running. Downloading and running MongoDB is pretty straightforward, see
    https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB for instructions.
    
    The parameter space to be searched over is specified in param_space. See https://github.com/hyperopt/hyperopt/wiki/FMin,
    section "Defining a search space" for further info. Also see the (very basic) example in contrib.rllab_hyperopt.example.main.py.
    
    NOTE: While the argument n_hyperopt_workers specifies the number of (local) parallel hyperopt workers to start, an equal
    number of EC2 instances will be started in parallel!
    NOTE2: Rllab currently terminates / starts a new EC2 instance for every task. This means what you'll pay amounts to
    hyperopt_max_evals * instance_hourly_rate. So you might want to be conservative with hyperopt_max_evals.
        
    :param task_method: the stubbed method call that runs the actual task. Should take a single dict as argument, with
        the params to evaluate. See e.g. contrib.rllab_hyperopt.example.task.py
    :param eval_method: the stubbed method call that reads in results returned from S3 and produces a score. Should take
        the exp_prefix and exp_name as arguments (this is where S3 results will be synced to). See e.g.
        contrib.rllab_hyperopt.example.score.py
    :param param_space: dict specifying the param space to search. See https://github.com/hyperopt/hyperopt/wiki/FMin,
        section "Defining a search space" for further info
    :param hyperopt_experiment_key: str, the key hyperopt will use to store results in the DB
    :param hyperopt_db_host: str, optional (default "localhost"). The host where mongodb runs
    :param hyperopt_db_port: int, optional (default 1234), the port where mongodb is listening for connections
    :param hyperopt_db_name: str, optional (default "rllab"), the DB name where hyperopt will store results
    :param n_hyperopt_workers: int, optional (default 1). The nr of parallel workers to start. NOTE: an equal number of
        EC2 instances will be started in parallel.
    :param hyperopt_max_evals: int, optional (defailt 100). Number of parameterset evaluations hyperopt should try.
        NOTE: Rllab currently terminates / starts a new EC2 instance for every task. This means what you'll pay amounts to
        hyperopt_max_evals * instance_hourly_rate. So you might want to be conservative with hyperopt_max_evals.
    :param result_timeout: int, optional (default 1200). Nr of seconds to wait for results from S3 for a given task. If
        results are not in within this time frame, <max_retries> new attempts will be made. A new attempt entails launching
        the task again on a new EC2 instance.
    :param max_retries: int, optional (default 0). Number of times to retry launching a task when results don't come in from S3
    :param run_experiment_kwargs: dict, optional (default None). Further kwargs to pass to run_experiment_lite. Note that
        specified values for exp_prefix, exp_name, variant, and confirm_remote will be ignored.
    :return the best result as found by hyperopt.fmin
    """
    exp_key = hyperopt_experiment_key
    
    worker_args = {'exp_prefix':exp_key,
                   'task_module':task_method.__module__,
                   'task_function':task_method.__name__,
                   'eval_module':eval_method.__module__,
                   'eval_function':eval_method.__name__,
                   'result_timeout':result_timeout,
                   'max_retries':max_retries}
          
    worker_args.update(param_space)
    if run_experiment_kwargs is not None:
        worker_args['run_experiment_kwargs'] = run_experiment_kwargs
     
    trials = MongoTrials('mongo://{0}:{1:d}/{2}/jobs'.format(hyperopt_db_host, hyperopt_db_port, hyperopt_db_name),
                         exp_key=exp_key)
     
    workers = _launch_workers(exp_key, n_hyperopt_workers, hyperopt_db_host, hyperopt_db_port, hyperopt_db_name)
     
    s3sync = S3SyncThread()
    s3sync.start()
     
    print("Starting hyperopt") 
    best = fmin(objective_fun, worker_args, trials=trials, algo=tpe.suggest, max_evals=hyperopt_max_evals)
         
    s3sync.stop()
    s3sync.join()
     
    for worker in workers:
        worker.terminate()
    
    return best
