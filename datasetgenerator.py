import pandas as pd
import numpy as np
import csv
df = pd.DataFrame(columns=['di', 'vi', 'a', 'kp', 'kd', 'ki'])
for k in range(1,126):
    new_row = []
    dataset = pd.read_csv("training_set_"+str(k)+".csv")
    for j in range(3):
        new_row.append(dataset.iloc[0,j])
    data = list()
    f = open("training_set_"+str(k)+"_positions_leading_car.csv",'r')
    rea = csv.reader(f)
    row_list = []
    for row in rea:
        for val in row:
            row_list.append(float(val))
        data.append(row_list)
        row_list = []
    f.close
    data = np.array(data)
    data = data.squeeze()
    min=0
    minarg=0
    for j in range(300):
        min = min + abs(data[j]-dataset.iloc[0,j+6])
    for i in range(1,3374):
        sum=0
        for j in range(299,279,-1):
            sum = sum + abs(20-abs(data[j]- dataset.iloc[i,j+6]))
        if sum>min:
            continue
        for j in range(280):
            sum = sum + abs(20-abs(data[j]- dataset.iloc[i,j+6]))
        if sum<min:
            min=sum
            minarg=i
    for j in range(3):
        new_row.append(dataset.iloc[minarg,j+3])
    df.loc[len(df)] = new_row
    print(k,new_row)
    df['ki'], df['kd'] = df['kd'], df['ki']
    df = df.rename(columns={'ki': 'kd', 'kd': 'ki'})
    df.to_csv('tuning.csv', index=False)