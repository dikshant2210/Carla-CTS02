import numpy as np
import pandas as pd
import pickle as pkl
import os

def process_log(filename=None):
    with open(filename, 'rb') as file:
        data = pkl.load(file)
        
    print("Scenario: {}".format(data[1]['scenario']))
    
    ttg = 0
    crash = 0
    nearmiss = 0
    exec_time = 0
    exec_count = 0
    for ep in data.keys():
        ttg += data[ep]['ttg']
        if data[ep]['crash']:
            crash += 1
        elif data[ep]['nearmiss']:
            nearmiss += 1
            
        count = 0
        temp = 0
        for t in data[ep]['exec']:
            if t != 0:
                temp += t
                count += 1
        if count != 0:
            exec_time += (temp / count)
            exec_count += 1
    
    ep_risk = []
    for key in data:
        count = 0
        total_ep_risk = 0
        for risk in data[key]['risk']:
            if risk != np.inf:
                count += 1
                total_ep_risk += risk
        ep_risk.append(total_ep_risk / count)
        
    sm = []
    action_dist = {0:0, 1:0, 2:0}
    for key in data:
        prev = 1
        count = 0
        for action in data[key]['actions']:
            action_dist[action] += 1
            if (prev == 0 and action == 2) or (prev == 2 and action == 1):
                count += 1
            prev = action
        sm.append(count)
    
    for key in action_dist:
        action_dist[key] /= len(data.keys())
    print(action_dist)
    
    ttg /= len(data.keys())
    crash /= len(data.keys())
    nearmiss /= len(data.keys())
    exec_time /= exec_count
    print("ttg: {:.2f}s".format(ttg))
    print("Crash: {:.2f}%".format(crash * 100))
    print("Nearmiss: {:.2f}%".format(nearmiss * 100))
    print("Risk: {:.3f}".format(sum(ep_risk) / len(ep_risk)))
    print("Sm: {:.3f}".format(sum(sm) / len(sm)))
    print("Exec. time: {:.2f}ms".format(exec_time * 1000))
    
directory = 'sac/pkl_res'
files = os.listdir(directory)

for f in files:
    try:
        process_log(filename=directory + '/{}'.format(f))
        print('------------------------')
    except:
        print(f)
        print('------------------------')
