# %%
import pandas as pd
import numpy as np

data = pd.read_csv('finds_data.csv')

concepts = np.array(data)[:,:-1]

target = np.array(data)[:,-1]

def train(con, tar):
    for i, val in enumerate(tar):
        if val == 'Yes':
            specific_h = con[i].copy()
            break
            
    for i, val in enumerate(con):
        if tar[i] == 'Yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
                else:
                    pass
        print(specific_h)
    
    print("Final Specific_h:")

    return specific_h

print(train(concepts, target))


# %%
