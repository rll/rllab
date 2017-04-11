import os
import pandas as pd

from rllab import config

def process_result(exp_prefix, exp_name):
    # Open the default rllab path for storing results
    result_path = os.path.join(config.LOG_DIR, "s3", exp_prefix, exp_name, 'progress.csv')
    print("Processing result from",result_path) 
    
    # This example uses pandas to easily read in results and create a simple smoothed learning curve
    df = pd.read_csv(result_path)
    curve = df['AverageReturn'].rolling(window=max(1,int(0.05*df.shape[0])), min_periods=1, center=True).mean().values.flatten()
    max_ix = curve.argmax()
    max_score = curve.max()
    
    # The result dict can contain arbitrary values, but ALWAYS needs to have a "loss" entry.
    return dict(
        max_score=max_score,
        max_iter=max_ix,
        scores=curve, # returning the curve allows you to plot best, worst etc curve later
        loss=-max_score
        )