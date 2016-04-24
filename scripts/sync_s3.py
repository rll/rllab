from rllab import config
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    args = parser.parse_args()
    remote_dir = config.AWS_S3_PATH
    local_dir = os.path.join(config.LOG_DIR, "s3")
    if args.folder:
        remote_dir = os.path.join(remote_dir, args.folder)
        local_dir = os.path.join(local_dir, args.folder)
    os.system("""
        aws s3 sync {remote_dir} {local_dir} --exclude '*debug.log' --exclude '*stdout.log' --exclude '*stdouterr.log' --exclude '*.pkl' --content-type "UTF-8"
    """.format(local_dir=local_dir, remote_dir=remote_dir))
