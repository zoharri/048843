import numpy as np

def main():
    filename = 'diabetes_scaled'
    #filename = 'breast_cancer_scaled'
    data_arr = []
    target_arr= []
    with open(filename+'.txt') as f:
        for line in f:
            data = line.split()
            target_arr.append(float(data[0])) # target value
            row = []
            for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
                n = int(idx) - (i + 1) # num missing
                for _ in range(n):
                    row.append(0) # for missing
                row.append(float(value))
            data_arr.append(row)

    data = np.array(data_arr)
    data = data/np.absolute(data).max()
    targets = np.array(target_arr) if filename == 'diabetes' else np.array(target_arr)-3
    np.save(filename+'_full',{'data': data,'targets':targets})

    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    targets = targets[indices]
    np.save(filename,{'train_data': [data[:300],targets[:300]],'val_data': [data[300:450],targets[300:450]],'test_data': [data[450:600],targets[450:600]]})

def suffle_and_split(dataset):
    blanced5=False
    while not blanced5:
        data = dataset['data']
        targets = dataset['targets']
        indices = np.random.permutation(data.shape[0])
        data = data[indices]
        targets = targets[indices]
        if targets[:5].sum()<2 and targets[:5].sum()>-2:
            blanced5=True
    return {'train_data': [data[:300],targets[:300]],'val_data': [data[300:450],targets[300:450]],'test_data': [data[450:600],targets[450:600]]}


if __name__ == "__main__":
    main()    
