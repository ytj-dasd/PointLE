import os
import numpy as np

batch_size = 15
n = 0
data = np.zeros((batch_size, 9))

f = open('merged_dataset/extra2/测试/4.txt')
fout = open(os.path.join('4.txt'), 'a')
line = f.readline()
while line:
    single_data = np.array(line.split('\t'))
    single_data[8] = single_data[8].strip('\n')
    #print(single_data)
    if n < 15:
        data[n] = single_data
    else:
        for i in range(batch_size-1):
            data[i] = data[i+1]
        data[batch_size-1] = single_data
    if n >= batch_size-1:
        for i in range(batch_size):
            for j in range(9):
                fout.write(str(data[i,j]) + ' ')
        fout.write('0 1 1\n')
    line = f.readline()
    n = n + 1

f.close()
fout.close()


#for instance, labels in test_dataloader: 
        #pred_val = model(instance)   
        #result = F.softmax(pred_val,dim=1)
        #output = torch.argmax(result,dim=1)
        #for j in range(len(instance)):
            #total = total +1
            #if output[j]==labels[j]:
                #good = good + 1    
    #return float(good / total)