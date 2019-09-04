#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Enter 5-tuple here


# In[ ]:


import numpy as np
c=np.asarray(L[0])
A=np.asarray(L[1])
b=np.asarray(L[2])
H=np.asarray(L[3])
d=np.asarray(L[4])


# In[ ]:


shapeA=A.shape
m1=shapeA[0]
n1=shapeA[1]
b1=[]                                     #creating a matrix of slack variables
I=[]                                     #Converting canonical LP into standard LP
for k in range(m1):
    a=[]
    for l in range(m1):
        if k==l:
            b1=1
        else:
            b1=0
        a.append(b)
    I.append(a)
I=np.asarray(I)

A=np.append(A,I,axis=1)
print(A)

shapeH=H.shape
m2=shapeH[0]
n2=shapeH[1]

Hzero=[]
for i2 in range(m2):
    a3=[]
    for j2 in range(m1):
        b3=0
        a3.append(b3)
    Hzero.append(a3)
    

H=np.append(H,Hzero,axis=1)
print(H)
M=np.append(A,H,axis=0)                        #appending A and H to form M
print(M)                                          
y=np.shape(M)
a=y[0]
b=y[1]
soln=np.append(b,d,axis=0)
print(soln)
print(a)
print(b)


# In[ ]:


slack=[]                         #Converting to ALP by adding an extra set of slack variables to M matrix
for i in range(a):               #Question 2(a)
    sl=[]
    for j in range(a):
        if i==j:
            sla=1
        else:
            sla=0
        sl.append(sla)
    slack.append(sl)
slack=np.asarray(slack)

print(slack)

M=np.append(M,slack,axis=1)     #Appending slack matrix to the M matrix
print(M)
shape=np.shape(M)
m=shape[0]
n=shape[1]
print(m)
print(n)


# 

# In[ ]:


for i in range(a):                   #Ensuring the values in solution matrix are all positive
    if soln[i]<0:
        soln[i]=-1*soln[i]
        M[i,:]=-1*M[i,:]
print(M)
print(soln)
        


# In[ ]:


#Formulating the Auxiliary Objective Function
c=[]
for i in range(n):
    if i>=b:
        c.append(1)
    else:
        c.append(0)
print(c)
    
    


# In[ ]:


#Formulating the First BFS
bfs=[]
for j in range(n):
    if j>=b:
        bfs.append(1)
    else:
        bfs.append(0)
print(bfs)


# In[1]:


def pivotbland(M,soln,c,bfs):               #Question 2(c)
    cn=[]                                   #Executing Phase 1 of Simplex
    cb=[]
    crindex=[]
    crindexb=[]
    N=[]
    B=[]
    count=0
    counter=0
    for i in range(n):                      #to segregate variables in Basic and Nonbasic matrices
        if bfs[i]==0:
            crindex.append(i)
            cn.append(c[i])
            N.append(M[:,i])
            count=count+1
        
        else: 
            crindexb.append(i)
            cb.append(c[i])
            B.append(M[:,i])
            counter=counter+1
        
    cb=np.asarray(cb)        # Cb
    cn=np.asarray(cn)        # Cn
    B=np.asarray(B)          # B
    B=B.transpose()          
    N=np.asarray(N)          # N
    N=N.transpose()
    

    
    cr=[]                                      #Finding the Reduced Cost Coefficient Vector(RCCV)
    Binv=np.linalg.inv(B)
    
    a=np.dot(Binv,N)
    b=np.dot(cb,a)
    cr=cn-b
    cr=np.asarray(cr)
    
    
    
    x=n-m                                    #Checking for optimality
    unbound=[]
    Ares=[]
    count=0 
    for j in cr:
        if j>=0:
            count=count+1
        if count==x:
            ofv1=np.matmul(cb,Binv)
            ofv=np.matmul(ofv1,soln)
            if ofv>0:
                y='infeasible'
            else:
                print('optimality reached at this BFS:', bfs ,'the Objective Function Value at this BFS is:',ofv)
                y='optimal'
            return y
    
    
    x=n-m                                      #checking for unboundedness
    check=[]
    kr=[]
    for k in range(x):
        if cr[k]>=0:
            kr=N[:,k].reshape((m,1))
            count=0
            unbound=np.matmul(Binv,kr)
            for j in unbound:
                if j<=0:
                    count=count+1
                    if count==m:
                        check.append(1)
            
    if len(check)>0:
        y='unbounded'
        return y
    
    p=list(cr)
    for i in range(x):
        if p[i]<0:
            l=i
            break
    v=crindex[l]             
    shiftn=M[:,v]                         # Choosing the NBV to be shifted to BV
    
    shiftn= np.asarray(shiftn)
    shiftn=shiftn.reshape((m,1))
    xb_start=np.matmul(Binv,soln)
    xb_middle=np.dot(Binv,shiftn)
    xb=[]
    xb=xb_start-xb_middle
    xb=np.asarray(xb)
    
    subtract=xb_start-xb                    # Choosing the BV to be shifted to NBV
    xbl=list(xb)
    for i in range(m):
        if subtract[i]>0 and xbl[i]==min(xbl):
            h=i
            break    
         
        else:
            continue
        
    j=crindexb[h]
    bfs=list(bfs)
    bfs[v]=1
    bfs[j]=0
    pivotbland(A,soln,c,bfs)


# In[2]:


pivotbland(M,soln,c,bfs)


# In[ ]:




