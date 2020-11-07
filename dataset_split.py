from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    root = '/data1/wangyanqing/data/webFG_5000/'

    fpath = []
    labels = []
    for d in os.listdir(root):
        fd = os.path.join(root, d)
        label = int(d)
        for i in os.listdir(fd):
            fp = os.path.join(fd, i)
            fpath.append(fp)
            labels.append(label)
            
    print(len(fpath), len(labels))
    
    x_train, x_val, y_train, y_val = train_test_split(fpath, labels, random_state=23, test_size=0.15)
    print(len(x_train), len(x_val))

    with open('train.txt', 'w')as f:
        for fn, l in zip(x_train, y_train):
            f.write('{} {}\n'.format(fn, l))

    with open('val.txt', 'w')as f:
        for fn, l in zip(x_val, y_val):
            f.write('{} {}\n'.format(fn, l))