#Finding nearest neighbour of query form 1M data

import numpy as np
import time

Actual_Ans_1M=[316385, 273688, 936940, 257478, 368156, 683745, 585955, 900884, 681310,  51238]

def hashGenr(arrayVector,normalVector):
    h_dot=np.dot(arrayVector,normalVector)
    h_dot=h_dot>0
    h_dot=h_dot.astype(int)
    return h_dot

#Loading of data as numpy array
arr=np.loadtxt("Data1M.csv", delimiter=",")


#Global VAr
arr_sortedSet=set()
listFinal=[]
nbits = 5  # no of vector
d = 100  # dimension of vevctor
No_of_hash=10
plane_norms_list=[]
LSH_buckets_list=[]


for i in range (No_of_hash):
    plane_norms = np.random.rand(d, nbits) - 0.5  # -0.5 help to complete axis
    plane_norms_list.append(plane_norms)
    v_dot = hashGenr(arr, plane_norms)

    # Buketing
    LSH_buckets = {}
    for i in range(len(v_dot)):
        hash_str = ''.join(v_dot[i].astype(str))
        if hash_str not in LSH_buckets.keys():
            LSH_buckets[hash_str] = set()
        LSH_buckets[hash_str].add(i)

    LSH_buckets_list.append(LSH_buckets)

t1=time.time()
qry = np.loadtxt("Q1.csv", delimiter=",")
for i in range(No_of_hash):
    qry = np.loadtxt("Q1.csv", delimiter=",")
    qry_dot = hashGenr(qry, plane_norms_list[i])
    qry_dot = ''.join(qry_dot.astype(str))  # converting from ndarray to string
    arr_sortedSet = arr_sortedSet.union(LSH_buckets_list[i][qry_dot])

print("------------LSH BUCKET List----------")
print(arr_sortedSet)
arr_sortedlist=list(arr_sortedSet)
tempArr = arr[arr_sortedlist]

res = (tempArr - qry) ** 2
d = res.sum(axis=1)
tempArr_sortedIndex = np.argsort(d, kind="mergesort")
print("\n-----Temp Sorted Arr Index----")
print(tempArr_sortedIndex)


print("\n-------------: :The index of data array for which query is nearest: :------\n")
for i in range(10):
    x = arr_sortedlist[tempArr_sortedIndex[i]]
    print(x)
    listFinal.append(x)

tf=time.time()

print(f"Exeution Time:{t1-tf}")
print("\n\n --------ans-------")
print(listFinal)

c=0
for i in range(10):
    if Actual_Ans_1M[i] in listFinal:
        c=c+1

print(f"The accuracy is:{c/10*100} %")