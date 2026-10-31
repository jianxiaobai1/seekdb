import sys, os, numpy as np
gt=np.fromfile(sys.argv[1],dtype='int32').reshape(-1,100)
out=np.fromfile(sys.argv[2],dtype='uint32').reshape(-1,100)
a10=sum(len(set(out[i,:10].tolist())&set(gt[i,:10].tolist()))/10 for i in range(len(gt)))/len(gt)
a100=sum(len(set(out[i,:100].tolist())&set(gt[i,:100].tolist()))/100 for i in range(len(gt)))/len(gt)
print('%s recall@10=%.4f recall@100=%.4f'%(os.path.basename(sys.argv[2]),a10,a100))
