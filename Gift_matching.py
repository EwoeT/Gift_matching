#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dataset as matrices
#define key values
import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph
import time

df_wish = np.array(pd.read_csv("child_wishlist_v2.csv", header=None))[:, 1:]
df_good = np.array(pd.read_csv("gift_goodkids_v2.csv", header=None))[:, 1:]
number_gifts = 1000
number_gift_types = 1000
number_children = 1000000
number_wishes = 100
number_goodkids = 1000

# df_good[:2]


# In[2]:



al = 1
al2 = 1

# function to calculate CH and GH
def ch_gh(gift_key, child_key, alpha):
    CH = alpha * (20 * (100 - gift_key))
    GH = (1-alpha)*(2 * (1000 - child_key))
    return CH, GH

# Create graph
start_nodes = []
end_nodes = []
capacities = []
costs = []
supplies = []

# graph from all gifts to special intermediate node
for gift in range(number_gift_types):
    supplies.append(number_gifts)
    
# supplies.append(0)

# graph from special intermediate node to all children
for child in range(number_children):
    if child<=5000:
        if child%3 == 0:
            supplies.append(-3)
        else:
            supplies.append(0)
    elif child >= 5001 and child <=45000:
        if child%2 == 1:
            supplies.append(-2)
        else:
            supplies.append(0)
    else:
        supplies.append(-1)
    
    
# graph from gifts to children    
for key in range(number_children):
    if key%10000 == 0:
        print((key/(len(df_wish)))*100,'%')
    for key_g in range(100):
        gift = df_wish[key, key_g]
        start_nodes.append(gift)
        end_nodes.append(1000+key)
        key_c = np.where(df_good[gift] == key)[0]
#         capacities.append(1)
#         for triplets
        if key<=5000:
            capacities.append(3)
            if key%3 == 0:
                for k in range(key,key+3):
                    k_g = np.where(df_wish[k] == gift)[0]
                    k_c = np.where(df_good[gift] == k)[0]
                    if k%3 == 0:
                        if k_c and k_g:
                            CH1, GH1 = ch_gh(k_g, k_c, al)
                        elif k_g:
                            CH1, GH1 = ch_gh(k_g, 1, al)
                            GH1 = 0
                        elif k_c:
                            CH1, GH1 = ch_gh(1, k_c, al)
                            CH1 = 0
                        else:
                            CH1=0
                            GH1=0
                    if k%3 == 1:
                        if k_c and k_g:
                            CH2, GH2 = ch_gh(k_g, k_c, al)
                        elif k_g:
                            CH2, GH2 = ch_gh(k_g, 1, al)
                            GH2 = 0
                        elif k_c:
                            CH2, GH2 = ch_gh(1, k_c, al)
                            CH2 = 0
                        else:
                            CH2=0
                            GH2=0
                    if k%3 == 2:
                        if k_c and k_g:
                            CH3, GH3 = ch_gh(k_g, k_c, al)
                        elif k_g:
                            CH3, GH3 = ch_gh(k_g, 1, al)
                            GH3 = 0
                        elif k_c:
                            CH3, GH3 = ch_gh(1, k_c, al)
                            CH3 = 0
                        else:
                            CH3=0
                            GH3=0
                CH = CH1+CH2+CH3
                GH = GH1+GH2+GH3
                costs.append(-(2)*(CH + GH))
            else:
                costs.append(1)
#                 for twins
        elif key >= 5001 and key <=45000:
            capacities.append(2)
            if key%2 == 1:
                for k in range(key,key+2):
                    k_g = np.where(df_wish[k] == gift)[0]
                    k_c = np.where(df_good[gift] == k)[0]
                    if k%2 == 1:
                        if k_c and k_g:
                            CH1, GH1 = ch_gh(k_g, k_c, al)
                        elif k_g:
                            CH1, GH1 = ch_gh(k_g, 1, al)
                            GH1 = 0
                        elif k_c:
                            CH1, GH1 = ch_gh(1, k_c, al)
                            CH1 = 0
                        else:
                            CH1=0
                            GH1=0
                    if k%2 == 0:
                        if k_c and k_g:
                            CH2, GH2 = ch_gh(k_g, k_c, al)
                        elif k_g:
                            CH2, GH2 = ch_gh(k_g, 1, al)
                            GH2 = 0
                        elif k_c:
                            CH2, GH2 = ch_gh(1, k_c, al)
                            CH2 = 0
                        else:
                            CH2=0
                            GH2=0

                CH = CH1+CH2
                GH = GH1+GH2
                costs.append(-(3)*(CH + GH))
            else:
                costs.append(1)
#         for single children
        else:
            capacities.append(1)
            if key_c:
                CH, GH = ch_gh(key_g, key_c, al)
                costs.append(-6*(CH + GH))
            else:
                CH, GH = ch_gh(key_g, 1, al)
                costs.append(-6*(CH))

for key in range(number_gift_types):
    if key%10 == 0:
        print((key/(len(df_good)))*100,'%')
    for key_c in range(1000):
        child = df_good[key,key_c]
        key_g = np.where(df_wish[child] == key)[0]
        if key_g:
            pass
        else:
          
            start_nodes.append(key)
            end_nodes.append(1000+child)   
            if child<=5000:
                capacities.append(3)
                if child%3 == 0:
                    CH1, GH1 = ch_gh(1, key_c, al2)
                elif child%3 == 1:
                    CH2, GH2 = ch_gh(1, key_c, al2)
                elif child%3 == 2:
                    CH3, GH3 = ch_gh(1, key_c, al2)
                    CH3 = 0
                    costs.append(-2*(GH1+GH2+GH3))
                    costs.append(1)
                    costs.append(1)
            if child>5000 and child <=45000:
                capacities.append(2)
                if child%2 == 1:
                    CH1, GH1 = ch_gh(1, key_c, al2)
                elif child%2 == 0:
                    CH2, GH2 = ch_gh(1, key_c, al2)
                    costs.append(-3*(GH1+GH2))
                    costs.append(1)
            if child>45000:
                    capacities.append(1)
                    CH, GH = ch_gh(1, key_c, al2)
                    costs.append(-6*(GH))

    
            
print('graph completed')              





# In[3]:


# Instantiate a SimpleMinCostFlow solver.
min_cost_flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc.
for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(int(start_nodes[i]), int(end_nodes[i]), int(capacities[i]), int(costs[i]))

# Add node supplies.

for i in range(0, len(supplies)):
    min_cost_flow.SetNodeSupply(i, supplies[i])


# In[4]:



gift_sharing = min_cost_flow.SolveMaxFlowWithMinCost()
print('Solved with status', gift_sharing, ' Pull results....')
print('Maximum flow:', min_cost_flow.MaximumFlow(), ' = ', number_children)


# In[5]:


# Solve minimum cost flow problem on graph input and get result_list of gift-child pair

result_list = []
for i in range(min_cost_flow.NumArcs()):
    if i % 5000000 == 0:
        print(i/1000000, "%")
    cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
    if cost != 0:
        re_tuple = (min_cost_flow.Tail(i),(min_cost_flow.Head(i)-1000))
        result_list.append(re_tuple)


# In[6]:



from collections import Counter
c = [i[1] for i in result_list]
countss = Counter(c)

duplicate_children = []
for k,v in countss.items():
    if v > 1:
        duplicate_children.append(k)

ss = []
for result in result_list:
    if result[1] in duplicate_children:
        if result[1] <5000:
            c_happiness = 0
            s_happiness = 0
            for rr in range(result[1], result[1]+3):
                try:
                    ch = np.where(df_wish[result[1]] == result[0])[0]
                    child_happ = 2*(100 - int(ch))
                except:
                    child_happ = -1
                try:
                    sh = np.where(df_good[result[0]] == result[1])[0]
                    santa_happ = 2*(1000 - int(sh))
                except:
                    santa_happ = -1
                c_h = child_happ/200
                s_h = santa_happ/2000
                c_happiness = c_happiness + c_h
                s_happiness = s_happiness + s_h
            score = c_happiness + s_happiness
            tup = (result[0], result[1], score)
            ss.append(tup)
        else:
            c_happiness = 0
            s_happiness = 0
            for rr in range(result[1], result[1]+2):
                try:
                    ch = np.where(df_wish[result[1]] == result[0])[0]
                    child_happ = 2*(100 - int(ch))
                except:
                    child_happ = -1
                try:
                    sh = np.where(df_good[result[0]] == result[1])[0]
                    santa_happ = 2*(1000 - int(sh))
                except:
                    santa_happ = -1
                c_h = child_happ/200
                s_h = santa_happ/2000
                c_happiness = c_happiness + c_h
                s_happiness = s_happiness + s_h
            score = c_happiness + s_happiness
            tup = (result[0], result[1], score)
            ss.append(tup)

for k,v in enumerate(ss[:-1]):
    if ss[k][1] == ss[k+1][1]:
        if ss[k][2]>ss[k+1][2]:
            ss[k+1] = v
        else:
            ss[k] = ss[k+1]
# ss= list(set(ss))
ss = [(i[0],i[1]) for i in list(set(ss))]
len(duplicate_children)


# In[7]:


# len(ss)
result_list = [i for i in result_list if i[1] not in duplicate_children]
len(result_list)


# In[8]:


for s in ss:
    result_list.append(s)
len(result_list)


# In[9]:


# get initial result_list
for result in result_list:
    if result[1]<= 5000:
        if result[1]%3 == 0:
            tup1=(result[0], result[1]+1)
            tup2=(result[0], result[1]+2)
            result_list.append(tup1)
            result_list.append(tup2)
    if result[1]>= 5001 and result[1]<= 45000:
        if result[1]%2 == 1:
            tup=(result[0], result[1]+1)
            result_list.append(tup)


# In[10]:


# check and collect excess gifts
import operator
from collections import Counter
excess_gifts = []
count_result2 = [i[0] for i in result_list]
count2 = Counter(count_result2)
count2 = sorted(count2.items(), key=operator.itemgetter(1), reverse=True)
for c in count2:
    if c[1]>1000:
        tup11 = (c[0], c[1]-1000)
        excess_gifts.append(tup11)
excess_gifts


# In[11]:


import operator
from collections import Counter
count_result3 = [i[0] for i in result_list]
count3 = Counter(count_result3)
count3 = sorted(count3.items(), key=operator.itemgetter(1))
# count3
less_gifts = []
for c in count3:
    if c[1]<1000:
        for i in range(1000-c[1]):
            less_gifts.append(c[0])
less_gifts


# In[12]:


# assign gifts to incomplete triplet or twin pairs by taking away from single children with that gift
for item in excess_gifts:
    cnt = 0
    for k,it in enumerate(result_list):
        if it[1]>45000 and it[0] == item[0] and cnt<item[1]:
            tuppt = (less_gifts.pop(), it[1])
            result_list[k] = tuppt
            cnt = cnt +1


# In[13]:


# count gifts in result_list
import operator
from collections import Counter
count_result4 = [i[0] for i in result_list]
count4 = Counter(count_result4)
count4 = sorted(count4.items(), key=operator.itemgetter(1), reverse=True)
count4


# In[14]:


# Evaluate the result (Happiness)

total_child_happiness = 0
nch = 0
total_santa_happiness = 0
nsh = 0

for result in result_list:
#     child_happiness = 0
    try:
        ch = np.where(df_wish[result[1]] == result[0])[0]
        child_happ = 2*(100 - int(ch))
    except:
        child_happ = -1
    try:
        sh = np.where(df_good[result[0]] == result[1])[0]
        santa_happ = 2*(1000 - int(sh))
    except:
        santa_happ = -1
    child_happiness = child_happ/200
    santa_happiness = santa_happ/2000
    total_child_happiness = total_child_happiness + child_happiness
    total_santa_happiness = total_santa_happiness + santa_happiness


nch = total_child_happiness/number_children
nsh = total_santa_happiness/(number_gift_types*number_gifts)
        
print("NCH: ",nch)
print("NSH: ",nsh)
print("total_happiness: ",(nch ** 3)+(nsh ** 3))

print("First evaluation done")        
 


# In[15]:


# Get lists of single children and their assigned gifts
from collections import Counter
count_list = []
new_gift_list = []
ng = []
new_children_list = []
# count_list = [result[0] for result in result_list if result[0] != 1000]
for result in result_list:
    if result[1] > 45000:
        count_list.append(result[0])
#     else:
#         new_children_list.append(result[1])
count = Counter(count_list)
for item, cnt in count.items():
    tup = (item, cnt)
    new_gift_list.append(tup)
    ng.append(item)
# len(new_children_list)
sum(count.values())  
# new_children_list


# In[16]:


new_gift_list.sort()
new_gift_list


# In[17]:


# Create graph of remaining gifts and unassigned children
start_nodes2 = []
end_nodes2 = []
capacities2 = []
costs2 = []
supplies2 = []



al = 0.998
al2 = 0.998

# function to calculate CH and GH
def ch_gh(gift_key, child_key, alpha):
    CH = 200000000 * (20 * (100 - gift_key))
    GH = (2)*(2 * (1000 - child_key))
    return CH, GH

# Create graph
start_nodes2 = []
end_nodes2 = []
capacities2 = []
costs2 = []
supplies2 = []

# graph from all gifts to special intermediate node
for gft in new_gift_list:
#     start_nodes.append(gift)
#     end_nodes.append(1000)
#     capacities.append(number_gifts)
#     costs.append(0)
    supplies2.append(gft[1])
    
# supplies.append(0)

# graph from special intermediate node to all children
for child in range(45001, 1000000):
#     start_nodes.append(1000)
#     end_nodes.append(child+1001)
#     capacities.append(1)
#     costs.append(1)
    supplies2.append(-1)
    
    
# graph from gifts to children    
for key in range(45001, number_children):
    if key%10000 == 0:
        print((key/(len(df_wish)))*100,'%')
    for key_g in range(100):
        gift = df_wish[key, key_g]
        start_nodes2.append(gift)
        end_nodes2.append(1000+(key-45001))
        key_c = np.where(df_good[gift] == key)[0]
        capacities2.append(1)
        if key_c:
            CH, GH = ch_gh(key_g, key_c, al)
            costs2.append(-1*(CH + GH))
        else:
            CH, GH = ch_gh(key_g, 1, al)
            costs2.append(-1*(CH))

for key in range(len(ng)):
    if key%10 == 0:
        print((key/(len(df_good)))*100,'%')
    for key_c in range(1000):
        child = df_good[key,key_c]
#         start_nodes.append(key)
#         end_nodes.append(1001+child)
#         capacities.append(1)
        key_g = np.where(df_wish[child] == key)[0]
        if key_g:
            pass
        else:
#             CH, GH = ch_gh(1, key_c, 1)
#             costs.append(-1*(GH))    
            if child>45000:
                start_nodes2.append(key)
                end_nodes2.append(1000+(child-45001))
                capacities2.append(1)
                CH, GH = ch_gh(1, key_c, al2)
                costs2.append(-1*(GH))

    
            
print('graph completed')        



    


# In[18]:


# solve new graph

# Instantiate a SimpleMinCostFlow solver.
min_cost_flow2 = pywrapgraph.SimpleMinCostFlow()

# Add each arc.
for i in range(0, len(start_nodes2)):
    min_cost_flow2.AddArcWithCapacityAndUnitCost(int(start_nodes2[i]), int(end_nodes2[i]), int(capacities2[i]), int(costs2[i]))

# Add node supplies.

for i in range(0, len(supplies2)):
    min_cost_flow2.SetNodeSupply(i, supplies2[i])


# In[19]:



gift_sharing2 = min_cost_flow2.SolveMaxFlowWithMinCost()
print('Solved with status', gift_sharing2, ' Pull results....')
print('Maximum flow:', min_cost_flow2.MaximumFlow(), ' = ', len(new_children_list))


# In[20]:


# solve new graph
result_list2 = []
for i in range(min_cost_flow2.NumArcs()):
    if i % 5000000 == 0:
        print(i)
    cost2 = min_cost_flow2.Flow(i) * min_cost_flow2.UnitCost(i)
#     if min_cost_flow.UnitCost(i) == 1:
#         print(i)
    if cost2 != 0:
#         for k1,v1 in enumerate(new_gift_list):
        ggift= min_cost_flow2.Tail(i)
#                 ggift = v1[0]
        cchild = (min_cost_flow2.Head(i)+45001) - 1000
#                 for k2,v2 in enumerate(new_children_list):
#                     if k2 == cchild_k:
#                         cchild = v2
        re_tuple2 = (ggift,cchild)
        result_list2.append(re_tuple2)


# In[21]:


result_list = [i for i in result_list if i[1]<=45000]
result_list = result_list + result_list2


# In[22]:


# Re-evaluate happiness

total_child_happiness = 0
nch = 0
total_santa_happiness = 0
nsh = 0

for result in result_list:
    try:
        ch = np.where(df_wish[result[1]] == result[0])[0]
        child_happ = 2*(100 - int(ch))
    except:
        child_happ = -1
    try:
        sh = np.where(df_good[result[0]] == result[1])[0]
        santa_happ = 2*(1000 - int(sh))
    except:
        santa_happ = -1
    child_happiness = child_happ/200
    santa_happiness = santa_happ/2000
    total_child_happiness = total_child_happiness + child_happiness
    
    total_santa_happiness = total_santa_happiness + santa_happiness


nch = total_child_happiness/number_children
nsh = total_santa_happiness/(number_gift_types*number_gifts)
        
print("NCH: ",nch)
print("NSH: ",nsh)
print("total_happiness: ",(nch ** 3)+(nsh ** 3))


print("Evaluation done")        


# In[18]:


# order result_list
def takeSecond(elem):
    return elem[1]

result_list.sort(key=takeSecond)


# In[19]:


import csv
writer = csv.writer(open("result.csv", 'w'))
header = ('ChildId,GiftId')
writer.writerow(header)
for row in result_list:
    writer.writerow(row[::-1])

