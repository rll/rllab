import os
import re
import subprocess
import base64
import os.path as osp
import pickle as pickle
import inspect
import hashlib
import sys
from contextlib import contextmanager

import errno

from rllab.core.serializable import Serializable
from rllab import config
from rllab.misc.console import mkdir_p
from rllab.misc import ext
from io import StringIO
import datetime
import dateutil.tz
import json
import time
import numpy as np

from rllab.misc.ext import AttrDict
from rllab.viskit.core import flatten
import collections


class StubBase(object):
    def __getitem__(self, item):
        return StubMethodCall(self, "__getitem__", args=[item], kwargs=dict())

    def __getattr__(self, item):
        try:
            return super(self.__class__, self).__getattribute__(item)
        except AttributeError:
            if item.startswith("__") and item.endswith("__"):
                raise
            return StubAttr(self, item)

    def __pow__(self, power, modulo=None):
        return StubMethodCall(self, "__pow__", [power, modulo], dict())

    def __call__(self, *args, **kwargs):
        return StubMethodCall(self.obj, self.attr_name, args, kwargs)

    def __add__(self, other):
        return StubMethodCall(self, "__add__", [other], dict())

    def __rmul__(self, other):
        return StubMethodCall(self, "__rmul__", [other], dict())

    def __div__(self, other):
        return StubMethodCall(self, "__div__", [other], dict())

    def __rdiv__(self, other):
        return StubMethodCall(BinaryOp(), "rdiv", [self, other], dict())  # self, "__rdiv__", [other], dict())

    def __rpow__(self, power, modulo=None):
        return StubMethodCall(self, "__rpow__", [power, modulo], dict())


class BinaryOp(Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())

    def rdiv(self, a, b):
        return b / a
        # def __init__(self, opname, a, b):
        #     self.opname = opname
        #     self.a = a
        #     self.b = b


class StubAttr(StubBase):
    def __init__(self, obj, attr_name):
        self.__dict__["_obj"] = obj
        self.__dict__["_attr_name"] = attr_name

    @property
    def obj(self):
        return self.__dict__["_obj"]

    @property
    def attr_name(self):
        return self.__dict__["_attr_name"]

    def __str__(self):
        return "StubAttr(%s, %s)" % (str(self.obj), str(self.attr_name))


class StubMethodCall(StubBase, Serializable):
    def __init__(self, obj, method_name, args, kwargs):
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.obj = obj
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return "StubMethodCall(%s, %s, %s, %s)" % (
            str(self.obj), str(self.method_name), str(self.args), str(self.kwargs))


class StubClass(StubBase):
    def __init__(self, proxy_class):
        self.proxy_class = proxy_class

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            # Convert the positional arguments to keyword arguments
            spec = inspect.getargspec(self.proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        return StubObject(self.proxy_class, *args, **kwargs)

    def __getstate__(self):
        return dict(proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError

    def __str__(self):
        return "StubClass(%s)" % self.proxy_class


class StubObject(StubBase):
    def __init__(self, __proxy_class, *args, **kwargs):
        if len(args) > 0:
            spec = inspect.getargspec(__proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        self.proxy_class = __proxy_class
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return dict(args=self.args, kwargs=self.kwargs, proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.args = dict["args"]
        self.kwargs = dict["kwargs"]
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        # why doesnt the commented code work?
        # return StubAttr(self, item)
        # checks bypassed to allow for accesing instance fileds
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError('Cannot get attribute %s from %s' % (item, self.proxy_class))

    def __str__(self):
        return "StubObject(%s, *%s, **%s)" % (str(self.proxy_class), str(self.args), str(self.kwargs))


class VariantDict(AttrDict):
    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator(object):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get("hide", False):
                self._hidden_keys.append(k)

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        methods = [x[1].__get__(self, self.__class__)
                   for x in methods if getattr(x[1], '__is_variant', False)]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        return list(map(self.variant_dict, ret))

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get("hide", False):
                suffix.append(k + "_" + str(variant[k]))
        return "_".join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, "__call__"):
                args = inspect.getargspec(vals).args
                if hasattr(vals, 'im_self') or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v)
                            for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getargspec(last_vals).args
                if hasattr(last_vals, 'im_self') or hasattr(last_vals, '__self__'):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(
                        **{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def variant(*args, **kwargs):
    def _variant(fn):
        fn.__is_variant = True
        fn.__variant_config = kwargs
        return fn

    if len(args) == 1 and isinstance(args[0], collections.Callable):
        return _variant(args[0])
    return _variant


def stub(glbs):
    # replace the __init__ method in all classes
    # hacky!!!
    for k, v in list(glbs.items()):
        # look at all variables that are instances of a class (not yet Stub)
        if isinstance(v, type) and v != StubClass:
            glbs[k] = StubClass(v)  # and replaces them by a the same but Stub


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
remote_confirmed = False


def run_experiment_lite(
        stub_method_call=None,
        batch_tasks=None,
        exp_prefix="experiment",
        exp_name=None,
        log_dir=None,
        script="scripts/run_experiment_lite.py",
        python_command="python",
        mode="local",
        dry=False,
        docker_image=None,
        aws_config=None,
        env=None,
        variant=None,
        use_gpu=False,
        sync_s3_pkl=False,
        sync_s3_png=False,
        sync_s3_log=False,
        sync_log_on_termination=True,
        confirm_remote=True,
        terminate_machine=True,
        periodic_sync=True,
        periodic_sync_interval=15,
        sync_all_data_node_to_s3=True,
        use_cloudpickle=None,
        pre_commands=None,
        added_project_directories=[],
        **kwargs):
    """
    Serialize the stubbed method call and run the experiment using the specified mode.
    :param stub_method_call: A stubbed method call.
    :param script: The name of the entrance point python script
    :param mode: Where & how to run the experiment. Should be one of "local", "local_docker", "ec2",
    and "lab_kube".
    :param dry: Whether to do a dry-run, which only prints the commands without executing them.
    :param exp_prefix: Name prefix for the experiments
    :param docker_image: name of the docker image. Ignored if using local mode.
    :param aws_config: configuration for AWS. Only used under EC2 mode
    :param env: extra environment variables
    :param kwargs: All other parameters will be passed directly to the entrance python script.
    :param variant: If provided, should be a dictionary of parameters
    :param use_gpu: Whether the launched task is running on GPU. This triggers a few configuration changes including
    certain environment flags
    :param sync_s3_pkl: Whether to sync pkl files during execution of the experiment (they will always be synced at
    the end of the experiment)
    :param sync_s3_png: Whether to sync png files during execution of the experiment (they will always be synced at
    the end of the experiment)
    :param sync_s3_log: Whether to sync log files during execution of the experiment (they will always be synced at
    the end of the experiment)
    :param confirm_remote: Whether to confirm before launching experiments remotely
    :param terminate_machine: Whether to terminate machine after experiment finishes. Only used when using
    mode="ec2". This is useful when one wants to debug after an experiment finishes abnormally.
    :param periodic_sync: Whether to synchronize certain experiment files periodically during execution.
    :param periodic_sync_interval: Time interval between each periodic sync, in seconds.
    """
    assert stub_method_call is not None or batch_tasks is not None, "Must provide at least either stub_method_call or batch_tasks"

    if use_cloudpickle is None:
        for maybe_stub in (batch_tasks or [stub_method_call]):
            # decide mode
            if isinstance(maybe_stub, StubBase):
                use_cloudpickle = False
            else:
                assert hasattr(maybe_stub, '__call__')
                use_cloudpickle = True
                # ensure variant exists
                if variant is None:
                    variant = dict()

    if batch_tasks is None:
        batch_tasks = [
            dict(
                kwargs,
                pre_commands=pre_commands,
                stub_method_call=stub_method_call,
                exp_name=exp_name,
                log_dir=log_dir,
                env=env,
                variant=variant,
                use_cloudpickle=use_cloudpickle
            )
        ]

    global exp_count
    global remote_confirmed
    config.USE_GPU = use_gpu

    # params_list = []

    for task in batch_tasks:
        call = task.pop("stub_method_call")
        if use_cloudpickle:
            import cloudpickle
            data = base64.b64encode(cloudpickle.dumps(call)).decode("utf-8")
        else:
            data = base64.b64encode(pickle.dumps(call)).decode("utf-8")
        task["args_data"] = data
        exp_count += 1
        params = dict(kwargs)
        if task.get("exp_name", None) is None:
            task["exp_name"] = "%s_%s_%04d" % (
                exp_prefix, timestamp, exp_count)
        if task.get("log_dir", None) is None:
            task["log_dir"] = config.LOG_DIR + "/local/" + \
                              exp_prefix.replace("_", "-") + "/" + task["exp_name"]
        if task.get("variant", None) is not None:
            variant = task.pop("variant")
            if "exp_name" not in variant:
                variant["exp_name"] = task["exp_name"]
            task["variant_data"] = base64.b64encode(pickle.dumps(variant)).decode("utf-8")
        elif "variant" in task:
            del task["variant"]
        task["remote_log_dir"] = osp.join(
            config.AWS_S3_PATH, exp_prefix.replace("_", "-"), task["exp_name"])
        task["env"] = task.get("env", dict()) or dict()
        task["env"]["RLLAB_USE_GPU"] = str(use_gpu)

    if mode not in ["local", "local_docker"] and not remote_confirmed and not dry and confirm_remote:
        remote_confirmed = query_yes_no(
            "Running in (non-dry) mode %s. Confirm?" % mode)
        if not remote_confirmed:
            sys.exit(1)

    if hasattr(mode, "__call__"):
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        mode(
            task,
            docker_image=docker_image,
            use_gpu=use_gpu,
            exp_prefix=exp_prefix,
            script=script,
            python_command=python_command,
            sync_s3_pkl=sync_s3_pkl,
            sync_log_on_termination=sync_log_on_termination,
            periodic_sync=periodic_sync,
            periodic_sync_interval=periodic_sync_interval,
            sync_all_data_node_to_s3=sync_all_data_node_to_s3,
        )
    elif mode == "local":
        for task in batch_tasks:
            del task["remote_log_dir"]
            env = task.pop("env", None)
            command = to_local_command(
                task,
                python_command=python_command,
                script=osp.join(config.PROJECT_PATH, script),
                use_gpu=use_gpu
            )
            print(command)
            if dry:
                return
            try:
                if env is None:
                    env = dict()
                subprocess.call(
                    command, shell=True, env=dict(os.environ, **env))
            except Exception as e:
                print(e)
                if isinstance(e, KeyboardInterrupt):
                    raise
    elif mode == "local_docker":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        for task in batch_tasks:
            del task["remote_log_dir"]
            env = task.pop("env", None)
            command = to_docker_command(
                task,  # these are the params. Pre and Post command can be here
                docker_image=docker_image,
                script=script,
                env=env,
                use_gpu=use_gpu,
                use_tty=True,
                python_command=python_command,
            )
            print(command)
            if dry:
                return
            p = subprocess.Popen(command, shell=True)
            try:
                p.wait()
            except KeyboardInterrupt:
                try:
                    print("terminating")
                    p.terminate()
                except OSError:
                    print("os error!")
                    pass
                p.wait()
    elif mode == "ec2":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        s3_code_path = s3_sync_code(config, dry=dry, added_project_directories=added_project_directories)
        launch_ec2(batch_tasks,
                   exp_prefix=exp_prefix,
                   docker_image=docker_image,
                   python_command=python_command,
                   script=script,
                   aws_config=aws_config,
                   dry=dry,
                   terminate_machine=terminate_machine,
                   use_gpu=use_gpu,
                   code_full_path=s3_code_path,
                   sync_s3_pkl=sync_s3_pkl,
                   sync_s3_png=sync_s3_png,
                   sync_s3_log=sync_s3_log,
                   sync_log_on_termination=sync_log_on_termination,
                   periodic_sync=periodic_sync,
                   periodic_sync_interval=periodic_sync_interval)
    elif mode == "lab_kube":
        # assert env is None
        # first send code folder to s3
        s3_code_path = s3_sync_code(config, dry=dry)
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        for task in batch_tasks:
            # if 'env' in task:
            #     assert task.pop('env') is None
            # TODO: dangerous when there are multiple tasks?
            task["resources"] = params.pop(
                "resources", config.KUBE_DEFAULT_RESOURCES)
            task["node_selector"] = params.pop(
                "node_selector", config.KUBE_DEFAULT_NODE_SELECTOR)
            task["exp_prefix"] = exp_prefix
            pod_dict = to_lab_kube_pod(
                task, code_full_path=s3_code_path, docker_image=docker_image, script=script, is_gpu=use_gpu,
                python_command=python_command,
                sync_s3_pkl=sync_s3_pkl, periodic_sync=periodic_sync,
                periodic_sync_interval=periodic_sync_interval,
                sync_all_data_node_to_s3=sync_all_data_node_to_s3,
                terminate_machine=terminate_machine,
            )
            pod_str = json.dumps(pod_dict, indent=1)
            if dry:
                print(pod_str)
            dir = "{pod_dir}/{exp_prefix}".format(
                pod_dir=config.POD_DIR, exp_prefix=exp_prefix)
            ensure_dir(dir)
            fname = "{dir}/{exp_name}.json".format(
                dir=dir,
                exp_name=task["exp_name"]
            )
            with open(fname, "w") as fh:
                fh.write(pod_str)
            kubecmd = "kubectl create -f %s" % fname
            print(kubecmd)
            if dry:
                return
            retry_count = 0
            wait_interval = 1
            while retry_count <= 5:
                try:
                    return_code = subprocess.call(kubecmd, shell=True)
                    if return_code == 0:
                        break
                    retry_count += 1
                    print("trying again...")
                    time.sleep(wait_interval)
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    print(e)
    else:
        raise NotImplementedError


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(params, python_command="python", script=osp.join(config.PROJECT_PATH,
                                                                      'scripts/run_experiment.py'),
                     use_gpu=False):
    command = python_command + " " + script
    if use_gpu and not config.USE_TF:
        command = "THEANO_FLAGS='device=gpu,dnn.enabled=auto,floatX=float32' " + command
    for k, v in config.ENV.items():
        command = ("%s=%s " % (k, v)) + command
    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)
    if pre_commands is not None or post_commands is not None:
        print("Not executing the pre_commands: ", pre_commands, ", nor post_commands: ", post_commands)

    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += \
                        "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    return command


def to_docker_command(params, docker_image, python_command="python", script='scripts/run_experiment_lite.py',
                      pre_commands=None, use_tty=False,
                      mujoco_path=None,
                      post_commands=None, dry=False, use_gpu=False, env=None, local_code_dir=None):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    docker_args = params.pop("docker_args", "")
    if pre_commands is None:
        pre_commands = params.pop("pre_commands", None)
    if post_commands is None:
        post_commands = params.pop("post_commands", None)
    if mujoco_path is None:
        mujoco_path = config.MUJOCO_KEY_PATH
    # script = 'rllab/' + script
    # if not dry:

    # create volume for logging directory
    if use_gpu:
        command_prefix = "nvidia-docker run"
    else:
        command_prefix = "docker run"
    docker_log_dir = config.DOCKER_LOG_DIR

    if env is None:
        env = dict()
    env = dict(
        env,
        AWS_ACCESS_KEY_ID=config.AWS_ACCESS_KEY,
        AWS_SECRET_ACCESS_KEY=config.AWS_ACCESS_SECRET,
    )
    if env is not None:
        for k, v in env.items():
            command_prefix += " -e \"{k}={v}\"".format(k=k, v=v)
    command_prefix += " -v {local_mujoco_key_dir}:{docker_mujoco_key_dir}".format(
        local_mujoco_key_dir=mujoco_path, docker_mujoco_key_dir='/root/.mujoco')
    command_prefix += " -v {local_log_dir}:{docker_log_dir}".format(
        local_log_dir=log_dir,
        docker_log_dir=docker_log_dir
    )
    command_prefix += docker_args
    if local_code_dir is None:
        local_code_dir = config.PROJECT_PATH
    command_prefix += " -v {local_code_dir}:{docker_code_dir}".format(
        local_code_dir=local_code_dir,
        docker_code_dir=config.DOCKER_CODE_DIR
    )
    params = dict(params, log_dir=docker_log_dir)
    if use_tty:
        command_prefix += " -ti " + docker_image + " /bin/bash -c "
    else:
        command_prefix += " -i " + docker_image + " /bin/bash -c "
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(to_local_command(
        params, python_command=python_command, script=osp.join(config.DOCKER_CODE_DIR, script), use_gpu=use_gpu))
    # We for 2 min sleep after termination to allow for last syncs.
    if post_commands is None:
        post_commands = ['sleep 120']
    command_list.extend(post_commands)
    return command_prefix + "'" + "; ".join(command_list) + "'"


def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)


def launch_ec2(params_list, exp_prefix, docker_image, code_full_path,
               python_command="python",
               script='scripts/run_experiment.py',
               aws_config=None, dry=False, terminate_machine=True, use_gpu=False, sync_s3_pkl=False,
               sync_s3_png=False,
               sync_s3_log=False,
               sync_log_on_termination=True,
               periodic_sync=True, periodic_sync_interval=15):
    if len(params_list) == 0:
        return

    default_config = dict(
        image_id=config.AWS_IMAGE_ID,
        instance_type=config.AWS_INSTANCE_TYPE,
        key_name=config.AWS_KEY_NAME,
        spot=config.AWS_SPOT,
        spot_price=config.AWS_SPOT_PRICE,
        iam_instance_profile_name=config.AWS_IAM_INSTANCE_PROFILE_NAME,
        security_groups=config.AWS_SECURITY_GROUPS,
        security_group_ids=config.AWS_SECURITY_GROUP_IDS,
        network_interfaces=config.AWS_NETWORK_INTERFACES,
    )

    if aws_config is None:
        aws_config = dict()
    aws_config = dict(default_config, **aws_config)

    sio = StringIO()
    sio.write("#!/bin/bash\n")
    sio.write("{\n")
    sio.write("""
        die() { status=$1; shift; echo "FATAL: $*"; exit $status; }
    """)
    sio.write("""
        EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"
    """)
    sio.write("""
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
    """.format(exp_name=params_list[0].get("exp_name"), aws_region=config.AWS_REGION_NAME))
    if config.LABEL:
        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=owner,Value={label} --region {aws_region}
        """.format(label=config.LABEL, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=exp_prefix,Value={exp_prefix} --region {aws_region}
    """.format(exp_prefix=exp_prefix, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        service docker start
    """)
    sio.write("""
        docker --config /home/ubuntu/.docker pull {docker_image}
    """.format(docker_image=docker_image))
    sio.write("""
        export AWS_DEFAULT_REGION={aws_region}
    """.format(aws_region=config.AWS_REGION_NAME))
    if config.FAST_CODE_SYNC:
        # sio.write("""
        #     aws s3 cp {code_full_path} /tmp/rllab_code.tar.gz --region {aws_region}
        # """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR,
        #            aws_region=config.AWS_REGION_NAME))
        sio.write("""
            aws s3 cp {code_full_path} /tmp/rllab_code.tar.gz
        """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR))
        sio.write("""
            mkdir -p {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR,
                   aws_region=config.AWS_REGION_NAME))
        sio.write("""
            tar -zxvf /tmp/rllab_code.tar.gz -C {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR,
                   aws_region=config.AWS_REGION_NAME))
    else:
        # sio.write("""
        #     aws s3 cp --recursive {code_full_path} {local_code_path} --region {aws_region}
        # """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR,
        #            aws_region=config.AWS_REGION_NAME))
        sio.write("""
            aws s3 cp --recursive {code_full_path} {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.DOCKER_CODE_DIR))

    s3_mujoco_key_path = config.AWS_CODE_SYNC_S3_PATH + '/.mujoco/'
    # sio.write("""
    #     aws s3 cp --recursive {} {} --region {}
    # """.format(s3_mujoco_key_path, config.MUJOCO_KEY_PATH, config.AWS_REGION_NAME))
    sio.write("""
        aws s3 cp --recursive {} {}
    """.format(s3_mujoco_key_path, config.MUJOCO_KEY_PATH))
    sio.write("""
        cd {local_code_path}
    """.format(local_code_path=config.DOCKER_CODE_DIR))

    for params in params_list:
        log_dir = params.get("log_dir")
        remote_log_dir = params.pop("remote_log_dir")
        env = params.pop("env", None)

        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
        """.format(exp_name=params.get("exp_name"), aws_region=config.AWS_REGION_NAME))
        sio.write("""
            mkdir -p {log_dir}
        """.format(log_dir=log_dir))
        if periodic_sync:
            include_png = " --include '*.png' " if sync_s3_png else " "
            include_pkl = " --include '*.pkl' " if sync_s3_pkl else " "
            include_log = " --include '*.log' " if sync_s3_log else " "
            # sio.write("""
            #     while /bin/true; do
            #         aws s3 sync --exclude '*' {include_png} {include_pkl} {include_log}--include '*.csv' --include '*.json' {log_dir} {remote_log_dir} --region {aws_region}
            #         sleep {periodic_sync_interval}
            #     done & echo sync initiated""".format(include_png=include_png, include_pkl=include_pkl, include_log=include_log,
            #                                          log_dir=log_dir, remote_log_dir=remote_log_dir,
            #                                          aws_region=config.AWS_REGION_NAME,
            #                                          periodic_sync_interval=periodic_sync_interval))
            sio.write("""
                while /bin/true; do
                    aws s3 sync --exclude '*' {include_png} {include_pkl} {include_log}--include '*.csv' --include '*.json' {log_dir} {remote_log_dir}
                    sleep {periodic_sync_interval}
                done & echo sync initiated""".format(include_png=include_png, include_pkl=include_pkl, include_log=include_log,
                                                     log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                     periodic_sync_interval=periodic_sync_interval))
            if sync_log_on_termination:
                # sio.write("""
                #     while /bin/true; do
                #         if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
                #           then
                #             logger "Running shutdown hook."
                #             aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log --region {aws_region}
                #             aws s3 cp --recursive {log_dir} {remote_log_dir} --region {aws_region}
                #             break
                #           else
                #             # Spot instance not yet marked for termination.
                #             sleep 5
                #         fi
                #     done & echo log sync initiated
                # """.format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
                sio.write("""
                    while /bin/true; do
                        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
                          then
                            logger "Running shutdown hook."
                            aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log
                            aws s3 cp --recursive {log_dir} {remote_log_dir}
                            break
                          else
                            # Spot instance not yet marked for termination.
                            sleep 5
                        fi
                    done & echo log sync initiated
                """.format(log_dir=log_dir, remote_log_dir=remote_log_dir))
        if use_gpu:
            sio.write("""
                for i in {1..800}; do su -c "nvidia-modprobe -u -c=0" ubuntu && break || sleep 3; done
                systemctl start nvidia-docker
            """)
        sio.write("""
            {command}
        """.format(command=to_docker_command(params, docker_image, python_command=python_command, script=script,
                                             use_gpu=use_gpu, env=env,
                                             local_code_dir=config.DOCKER_CODE_DIR)))
        # sio.write("""
        #     aws s3 cp --recursive {log_dir} {remote_log_dir} --region {aws_region}
        # """.format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
        sio.write("""
            aws s3 cp --recursive {log_dir} {remote_log_dir}
        """.format(log_dir=log_dir, remote_log_dir=remote_log_dir))
        # sio.write("""
        #     aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log --region {aws_region}
        # """.format(remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
        sio.write("""
            aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log
        """.format(remote_log_dir=remote_log_dir))

    if terminate_machine:
        sio.write("""
            EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
            aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region {aws_region}
        """.format(aws_region=config.AWS_REGION_NAME))
    sio.write("} >> /home/ubuntu/user_data.log 2>&1\n")

    full_script = dedent(sio.getvalue())

    import boto3
    import botocore
    if aws_config["spot"]:
        ec2 = boto3.client(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
    else:
        ec2 = boto3.resource(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )

    if len(full_script) > 10000 or len(base64.b64encode(full_script.encode()).decode("utf-8")) > 10000:
        # Script too long; need to upload script to s3 first.
        # We're being conservative here since the actual limit is 16384 bytes
        s3_path = upload_file_to_s3(full_script)
        sio = StringIO()
        sio.write("#!/bin/bash\n")
        sio.write("""
        aws s3 cp {s3_path} /home/ubuntu/remote_script.sh --region {aws_region} && \\
        chmod +x /home/ubuntu/remote_script.sh && \\
        bash /home/ubuntu/remote_script.sh
        """.format(s3_path=s3_path, aws_region=config.AWS_REGION_NAME))
        user_data = dedent(sio.getvalue())
    else:
        user_data = full_script
    print(full_script)
    with open("/tmp/full_script", "w") as f:
        f.write(full_script)

    instance_args = dict(
        ImageId=aws_config["image_id"],
        KeyName=aws_config["key_name"],
        UserData=user_data,
        InstanceType=aws_config["instance_type"],
        EbsOptimized=config.EBS_OPTIMIZED,
        SecurityGroups=aws_config["security_groups"],
        SecurityGroupIds=aws_config["security_group_ids"],
        NetworkInterfaces=aws_config["network_interfaces"],
        IamInstanceProfile=dict(
            Name=aws_config["iam_instance_profile_name"],
        ),
        **config.AWS_EXTRA_CONFIGS
    )

    if len(instance_args["NetworkInterfaces"]) > 0:
        # disable_security_group = query_yes_no(
        #     "Cannot provide both network interfaces and security groups info. Do you want to disable security group settings?",
        #     default="yes",
        # )
        disable_security_group = True
        if disable_security_group:
            instance_args.pop("SecurityGroups")
            instance_args.pop("SecurityGroupIds")

    if aws_config.get("placement", None) is not None:
        instance_args["Placement"] = aws_config["placement"]
    if not aws_config["spot"]:
        instance_args["MinCount"] = 1
        instance_args["MaxCount"] = 1
    print("************************************************************")
    print(instance_args["UserData"])
    print("************************************************************")
    if aws_config["spot"]:
        instance_args["UserData"] = base64.b64encode(instance_args["UserData"].encode()).decode("utf-8")
        spot_args = dict(
            DryRun=dry,
            InstanceCount=1,
            LaunchSpecification=instance_args,
            SpotPrice=aws_config["spot_price"],
            # ClientToken=params_list[0]["exp_name"],
        )
        import pprint
        pprint.pprint(spot_args)
        if not dry:
            response = ec2.request_spot_instances(**spot_args)
            print(response)
            spot_request_id = response['SpotInstanceRequests'][
                0]['SpotInstanceRequestId']
            for _ in range(10):
                try:
                    ec2.create_tags(
                        Resources=[spot_request_id],
                        Tags=[
                            {'Key': 'Name', 'Value': params_list[0]["exp_name"]}
                        ],
                    )
                    break
                except botocore.exceptions.ClientError:
                    continue
    else:
        import pprint
        pprint.pprint(instance_args)
        ec2.create_instances(
            DryRun=dry,
            **instance_args
        )


S3_CODE_PATH = None


def s3_sync_code(config, dry=False, added_project_directories=[]):
    global S3_CODE_PATH
    if S3_CODE_PATH is not None:
        return S3_CODE_PATH
    base = config.AWS_CODE_SYNC_S3_PATH
    has_git = True

    if config.FAST_CODE_SYNC:
        try:
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        except subprocess.CalledProcessError as _:
            print("Warning: failed to execute git commands")
            current_commit = None

        file_name = str(timestamp) + "_" + hashlib.sha224(
            subprocess.check_output(["pwd"]) + str(current_commit).encode() + str(timestamp).encode()
        ).hexdigest() + ".tar.gz"

        file_path = "/tmp/" + file_name

        tar_cmd = ["tar", "-zcvf", file_path, "-C", config.PROJECT_PATH]

        for pattern in config.FAST_CODE_SYNC_IGNORES:
            tar_cmd += ["--exclude", pattern]
        tar_cmd += ["-h", "."]

        for path in added_project_directories:
            tar_cmd.append("-C")
            tar_cmd.append(path)
            tar_cmd += ["."]

        remote_path = "%s/%s" % (base, file_name)

        upload_cmd = ["aws", "s3", "cp", file_path, remote_path]

        mujoco_key_cmd = [
            "aws", "s3", "sync", config.MUJOCO_KEY_PATH, "{}/.mujoco/".format(base)]

        print(" ".join(tar_cmd))
        print(" ".join(upload_cmd))
        print(" ".join(mujoco_key_cmd))

        if not dry:
            subprocess.check_call(tar_cmd)
            subprocess.check_call(upload_cmd)
            try:
                subprocess.check_call(mujoco_key_cmd)
            except Exception as e:
                print(e)

        S3_CODE_PATH = remote_path
        return remote_path
    else:
        try:
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            clean_state = len(
                subprocess.check_output(["git", "status", "--porcelain"])) == 0
        except subprocess.CalledProcessError as _:
            print("Warning: failed to execute git commands")
            has_git = False
        dir_hash = base64.b64encode(subprocess.check_output(["pwd"])).decode("utf-8")
        code_path = "%s_%s" % (
            dir_hash,
            (current_commit if clean_state else "%s_dirty_%s" % (current_commit, timestamp)) if
            has_git else timestamp
        )
        full_path = "%s/%s" % (base, code_path)
        cache_path = "%s/%s" % (base, dir_hash)
        cache_cmds = ["aws", "s3", "cp", "--recursive"] + \
                     flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
                     [cache_path, full_path]
        cmds = ["aws", "s3", "cp", "--recursive"] + \
               flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
               [".", full_path]
        caching_cmds = ["aws", "s3", "cp", "--recursive"] + \
                       flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
                       [full_path, cache_path]
        mujoco_key_cmd = [
            "aws", "s3", "sync", config.MUJOCO_KEY_PATH, "{}/.mujoco/".format(base)]
        print(cache_cmds, cmds, caching_cmds, mujoco_key_cmd)
        if not dry:
            subprocess.check_call(cache_cmds)
            subprocess.check_call(cmds)
            subprocess.check_call(caching_cmds)
            try:
                subprocess.check_call(mujoco_key_cmd)
            except Exception:
                print('Unable to sync mujoco keys!')
        S3_CODE_PATH = full_path
        return full_path


def upload_file_to_s3(script_content):
    import tempfile
    import uuid
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(script_content.encode())
    f.close()
    remote_path = os.path.join(
        config.AWS_CODE_SYNC_S3_PATH, "oversize_bash_scripts", str(uuid.uuid4()))
    subprocess.check_call(["aws", "s3", "cp", f.name, remote_path])
    os.unlink(f.name)
    return remote_path


def to_lab_kube_pod(
        params, docker_image, code_full_path,
        python_command="python",
        script='scripts/run_experiment.py',
        is_gpu=False,
        sync_s3_pkl=False,
        periodic_sync=True,
        periodic_sync_interval=15,
        sync_all_data_node_to_s3=False,
        terminate_machine=True
):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    remote_log_dir = params.pop("remote_log_dir")
    resources = params.pop("resources")
    node_selector = params.pop("node_selector")
    exp_prefix = params.pop("exp_prefix")

    kube_env = [
        {"name": k, "value": v}
        for k, v in (params.pop("env", None) or dict()).items()
        ]
    mkdir_p(log_dir)
    pre_commands = list()
    pre_commands.append('mkdir -p ~/.aws')
    pre_commands.append('mkdir ~/.mujoco')
    # fetch credentials from the kubernetes secret file
    pre_commands.append('echo "[default]" >> ~/.aws/credentials')
    pre_commands.append(
        "echo \"aws_access_key_id = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_KEY)
    pre_commands.append(
        "echo \"aws_secret_access_key = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_SECRET)
    s3_mujoco_key_path = config.AWS_CODE_SYNC_S3_PATH + '/.mujoco/'
    pre_commands.append(
        'aws s3 cp --recursive {} {}'.format(s3_mujoco_key_path, '~/.mujoco'))

    if config.FAST_CODE_SYNC:
        pre_commands.append('aws s3 cp %s /tmp/rllab_code.tar.gz' % code_full_path)
        pre_commands.append('mkdir -p %s' % config.DOCKER_CODE_DIR)
        pre_commands.append('tar -zxvf /tmp/rllab_code.tar.gz -C %s' % config.DOCKER_CODE_DIR)
    else:
        pre_commands.append('aws s3 cp --recursive %s %s' %
                            (code_full_path, config.DOCKER_CODE_DIR))
    pre_commands.append('cd %s' % config.DOCKER_CODE_DIR)
    pre_commands.append('mkdir -p %s' %
                        (log_dir))

    if sync_all_data_node_to_s3:
        print('Syncing all data from node to s3.')
        if periodic_sync:
            if sync_s3_pkl:
                pre_commands.append("""
                            while /bin/true; do
                                aws s3 sync {log_dir} {remote_log_dir} --region {aws_region} --quiet
                                sleep {periodic_sync_interval}
                            done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                                 aws_region=config.AWS_REGION_NAME,
                                                                 periodic_sync_interval=periodic_sync_interval))
            else:
                pre_commands.append("""
                            while /bin/true; do
                                aws s3 sync {log_dir} {remote_log_dir} --region {aws_region} --quiet
                                sleep {periodic_sync_interval}
                            done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                                 aws_region=config.AWS_REGION_NAME,
                                                                 periodic_sync_interval=periodic_sync_interval))
    else:
        if periodic_sync:
            if sync_s3_pkl:
                pre_commands.append("""
                    while /bin/true; do
                        aws s3 sync --exclude '*' --include '*.csv' --include '*.json' --include '*.pkl' {log_dir} {remote_log_dir} --region {aws_region} --quiet
                        sleep {periodic_sync_interval}
                    done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                         aws_region=config.AWS_REGION_NAME,
                                                         periodic_sync_interval=periodic_sync_interval))
            else:
                pre_commands.append("""
                    while /bin/true; do
                        aws s3 sync --exclude '*' --include '*.csv' --include '*.json' {log_dir} {remote_log_dir} --region {aws_region} --quiet
                        sleep {periodic_sync_interval}
                    done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                         aws_region=config.AWS_REGION_NAME,
                                                         periodic_sync_interval=periodic_sync_interval))
    # copy the file to s3 after execution
    post_commands = list()
    post_commands.append('aws s3 cp --recursive %s %s' %
                         (log_dir,
                          remote_log_dir))
    if not terminate_machine:
        post_commands.append('sleep infinity')
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(
        "%s 2>&1 | tee -a %s" % (
            to_local_command(params, python_command=python_command, script=script),
            "%s/stdouterr.log" % log_dir
        )
    )
    if post_commands is not None:
        command_list.extend(post_commands)
    command = "; ".join(command_list)
    pod_name = config.KUBE_PREFIX + params["exp_name"]
    # underscore is not allowed in pod names
    pod_name = pod_name.replace("_", "-")
    print("Is gpu: ", is_gpu)
    if not is_gpu:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "labels": {
                    "owner": config.LABEL,
                    "expt": pod_name,
                    "exp_time": timestamp,
                    "exp_prefix": exp_prefix,
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "foo",
                        "image": docker_image,
                        "command": [
                            "/bin/bash",
                            "-c",
                            "-li",  # to load conda env file
                            command,
                        ],
                        "resources": resources,
                        "imagePullPolicy": "Always",
                    }
                ],
                "restartPolicy": "Never",
                "nodeSelector": node_selector,
                "dnsPolicy": "Default",
            }
        }
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "owner": config.LABEL,
                "expt": pod_name,
                "exp_time": timestamp,
                "exp_prefix": exp_prefix,
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "foo",
                    "image": docker_image,
                    "env": kube_env,
                    "command": [
                        "/bin/bash",
                        "-c",
                        "-li",  # to load conda env file
                        command,
                    ],
                    "resources": resources,
                    "imagePullPolicy": "Always",
                    # gpu specific
                    "volumeMounts": [
                        {
                            "name": "nvidia",
                            "mountPath": "/usr/local/nvidia",
                            "readOnly": True,
                        }
                    ],
                    "securityContext": {
                        "privileged": True,
                    }
                }
            ],
            "volumes": [
                {
                    "name": "nvidia",
                    "hostPath": {
                        "path": "/var/lib/docker/volumes/nvidia_driver_352.63/_data",
                    }
                }
            ],
            "restartPolicy": "Never",
            "nodeSelector": node_selector,
            "dnsPolicy": "Default",
        }
    }


def concretize(maybe_stub):
    if isinstance(maybe_stub, StubMethodCall):
        obj = concretize(maybe_stub.obj)
        method = getattr(obj, maybe_stub.method_name)
        args = concretize(maybe_stub.args)
        kwargs = concretize(maybe_stub.kwargs)
        return method(*args, **kwargs)
    elif isinstance(maybe_stub, StubClass):
        return maybe_stub.proxy_class
    elif isinstance(maybe_stub, StubAttr):
        obj = concretize(maybe_stub.obj)
        attr_name = maybe_stub.attr_name
        attr_val = getattr(obj, attr_name)
        return concretize(attr_val)
    elif isinstance(maybe_stub, StubObject):
        if not hasattr(maybe_stub, "__stub_cache"):
            args = concretize(maybe_stub.args)
            kwargs = concretize(maybe_stub.kwargs)
            try:
                maybe_stub.__stub_cache = maybe_stub.proxy_class(
                    *args, **kwargs)
            except Exception as e:
                print(("Error while instantiating %s" % maybe_stub.proxy_class))
                import traceback
                traceback.print_exc()
        ret = maybe_stub.__stub_cache
        return ret
    elif isinstance(maybe_stub, dict):
        # make sure that there's no hidden caveat
        ret = dict()
        for k, v in maybe_stub.items():
            ret[concretize(k)] = concretize(v)
        return ret
    elif isinstance(maybe_stub, (list, tuple)):
        return maybe_stub.__class__(list(map(concretize, maybe_stub)))
    else:
        return maybe_stub
