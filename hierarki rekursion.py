# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

# %%
df = pd.DataFrame(data={'id':[1.0,2.0,3.0,4.0,5.0],'parent':[np.NaN,1,2,1,3]})

# %%
df_fasit = pd.DataFrame(data={'id':[1.0,2.0,3.0,4.0], 'nivå_1':[1,1,1,1], 'nivå_2':[1,2,3,4], 'nivå_3':[1,2,3,4]})


# %%
def children(parents, i, df_out):
    temp = pd.DataFrame()
    for _,parent in enumerate(parents):
        df_child = df[df.parent == parent]
        df_parent = df[df.id == parent]
        if df_child.shape[0] == 0:
            pass
        else:
            temp['id_'+str(i)] = df_child.loc[:, ('parent')]
            temp['id_'+str(i+1)] = df_child.loc[:, ('id')]
#             child_merge = df_child.merge(df_parent, left_on='id', right_on='parent')
            temp = pd.concat([temp, df_child])
            grand = children(parents=df_child.id, i = i+1, df_out=temp)
            temp = pd.concat([temp, grand])
    df_out = pd.concat([df_out, temp])
    return  df_out
    
parent = [1.0]#df_parent.id.values
df_out = pd.DataFrame()
test = children(parents=parent, i=0, df_out=df_out)
test.head()

