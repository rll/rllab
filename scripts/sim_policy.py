from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import os
import random
import numpy as np
import tensorflow as tf

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    policy = None
    env = None

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    while True:
        with tf.Session() as sess:
            data = joblib.load(args.file)
            policy = data['policy']
            env = data['env']
            while True:
                path = rollout(env, policy, max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup)
