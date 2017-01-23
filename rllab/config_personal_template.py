import os

USE_GPU = False

DOCKER_IMAGE = "dementrock/rllab3-shared"

KUBE_PREFIX = "template_"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_IMAGE_ID = "ami-67c5d00d"

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

AWS_KEY_NAME = "research_virginia"

AWS_SPOT = True

AWS_SPOT_PRICE = '10.0'

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab"]

AWS_REGION_NAME = "us-west-2"

AWS_CODE_SYNC_S3_PATH = "<insert aws s3 bucket url for code>e"

CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*src/*",
                     "*.pods/*", "*tests/*", "*examples/*", "docs/*"]

LOCAL_CODE_DIR = "<insert local code dir>"

AWS_S3_PATH = "<insert aws s3 bucket url>"

LABEL = "template"

DOCKER_CODE_DIR = "/root/code/rllab"

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", "<insert aws key>")

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", "<insert aws secret>")
