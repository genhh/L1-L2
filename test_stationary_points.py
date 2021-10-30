import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import shape
from numpy.linalg.linalg import cholesky, norm, svd,inv
import matplotlib.pyplot as plt
import time
import math

def Ps(x,pattern):
    #define Ps operator：y=sign(x)
    y = 0*x
    y[np.where(pattern>0)[0]]=1
    y[np.where(pattern<0)[0]]=-1
    idm = np.where(pattern==0)[0]
    temp = np.stack((x[idm].reshape(-1,1),np.ones((idm.shape[0],1))),axis=1)
    temp = np.min(temp,axis=1)
    temp = np.stack((temp,-np.ones((idm.shape[0],1))),axis=1)
    y[idm]=np.max(temp,axis=1) #在pattern为0的位置的所有元素找出最小一批中的最大
    return y

def dct(m,n):
    # dct matrix
    C_temp = np.zeros(m,n)
    C_temp[0, :] = 1 * np.sqrt(1/n)
    for i in range(1, m):
     for j in range(n):
          C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * n )) * np.sqrt(2 / n )
    return C_temp

def construct_b(A,x,lam=1e-2):
    #init
    m,n = A.shape
    maxiter = 10*n
    tolres = 1e-10
    verbose = False
    U,sigma,VT=np.linalg.svd(A.T) #U对应标准正交基
    rank = np.linalg.matrix_rank(A.T)
    U = U[:,:rank]
    w0 = np.zeros([n,1])
    w0[x>0] = 1
    w0[x<0] = -1
    w = w0
    x2 = x/norm(x, ord=2, axis=None, keepdims=False)
    err = []
    for i in range(maxiter):
        w_old = w
        v = U@U.T@(w_old-x2)
        error_rgAt = norm(v-w_old+x2,ord=2)
        w = Ps(v,w0)
        err.append(error_rgAt)
        
        if error_rgAt < tolres:
            break
    y = inv(A@A.T)@A@(w-x2)
    b = lam*y + A@x
    err = np.array(err)
    return b

def construct(m,n,mtype,k=10):
    # construct A,xg,b
    if mtype==1:
        A = np.random.randn(m,n)
        A = A / norm(A,ord=2)
    elif mtype==2:
        A = dct(n)
        idx = np.array(range(n-1))
        idx = np.random.randn.shuffle(idx) + 2
        A = A[[1,idx[0:m-1]],:]
        A = A /norm(A,ord=2)
    x_g = np.zeros([n,1])
    xs = np.random.randn(k,1)
    idn = np.random.choice(a=n, size=k, replace=False, p=None)
    x_g[idn] = xs
    b = construct_b(A,x_g)
    return A,b,x_g


def shrinkL12(y,lam,alpha=1):
    #proxy算子?
    x = np.zeros(y.shape)
    if max(abs(y)) > 0: #have non-zeros elements
        if max(abs(y)) > lam:
            temp = np.stack((abs(y) - lam, np.zeros(y.shape)),axis=1)
            x = np.max(temp,axis=1)*np.sign(y)
            x = x*(norm(x) + alpha*lam)/norm(x)
        elif max(abs(y)) >= (1-alpha)*lam:
                i =np.where(abs(y)==max(abs(y)))  #i表示坐标索引
                x[i[0]] = (abs(y[i[0]])+(alpha-1)*lam)*np.sign(y[i[0]])
    return x

def shrink(x,r):
    temp = np.stack((abs(x)-r,np.zeros(x.shape)),axis=1)
    z = np.sign(x)*np.max(temp,axis=1)
    return z

def L1L2_ADMM(A,b,x_g,fuc=shrinkL12,lam=1e-5,alpha=1,reltol=1e-6):
    m,n = A.shape
    maxit = n*5
    x0 = np.zeros([n,1])
    delta = 100*lam
    start = time.time()
    
    AAt = A@A.T
    L = cholesky(np.eye(m)+1/delta*AAt)
    U = L.T
    x = np.zeros([n,1])
    Atb = A.T@b
    y = np.zeros([n,1])
    u = x
    eps = 2.2204e-16

    obj = lambda x:0.5*norm(A@x-b)**2 + lam*(norm(x,ord=1)-alpha*norm(x))
    relall = []
    for i in range(maxit):
        x = fuc(y - u, lam/delta)  #与原ADMM区别在这里
        if i==0:
            x = x0
        y_old = y
        rhs = Atb + delta*(x+u)
        y = rhs/delta - (A.T@(inv(U.T@U)@U.T@(inv(U@U.T)@U@(A@rhs))))/delta**2
        u = u + x - y
        relerr = norm(y_old-y)/max(norm(y_old),norm(y),eps)
        err = norm(x-x_g)/norm(x_g)
        relall.append(err)
        if relerr<reltol and i>2:
            break
    end = time.time()
    cost = end -start
    return cost,relall

def L1L2_DCA(A,b,x_g,fuc=shrink,lam=1e-5,alpha=1,reltol=1e-6):
    m,n = A.shape
    maxit = n*5
    x0 = np.zeros([n,1])
    delta = 100*lam
    start = time.time()
    
    AAt = A@A.T
    L = cholesky(np.eye(m)+1/delta*AAt)
    U = L.T
    x = np.zeros([n,1])
    Atb = A.T@b
    y = np.zeros([n,1])
    u = x
    eps = 2.2204e-16

    obj = lambda x:0.5*norm(A@x-b)**2 + lam*(norm(x,ord=1)-alpha*norm(x))
    relall = []
    maxot = 10
    for j in range(maxot):
        c = alpha*x/(norm(x,ord=2)+eps) #difference
        xold = x
        for i in range(maxit):
            xoldinner = x
            x = fuc(y - u, lam/delta)  #与原ADMM区别在这里
            
            y_old = y
            rhs = Atb + delta*(x+u) +lam*c
            y = rhs/delta - (A.T@(inv(U.T@U)@U.T@(inv(U@U.T)@U@(A@rhs))))/delta**2
            u = u + x - y
            relerr = norm(x-y)/max(norm(x),norm(y),eps) #difference
            err = norm(x-x_g)/norm(x_g)
            relall.append(err)
            relerrinner = relerr
            if relerr<reltol: #difference
                break
        relerr = np.sqrt(sum((x-xold)**2))/max(np.sqrt(sum(x**2)),1) #difference
        if relerr<reltol: #difference
            break
    end = time.time()
    cost = end -start
    return cost,relall
    
def L1L2_FBS(A,b,x_g,fuc=shrinkL12,lam=1e-5,alpha=1,reltol=1e-6):
    m,n = A.shape
    maxit = n*5
    x0 = np.zeros([n,1])
    delta = 1 #difference
    start = time.time()
    
    AAt = A@A.T
    x = np.zeros([n,1])
    y = np.zeros([n,1])
    z = np.zeros([n,1])
    x_old = np.zeros([n,1])
    t = 1
    t_old = 1
    eps = 2.2204e-16

    obj = lambda x:0.5*norm(A@x-b)**2 + lam*(norm(x,ord=1)-alpha*norm(x))
    relall = []
    for i in range(maxit):
        y = x + t_old/t * (z-x) + (t_old-1)/t *(x-x_old)
        z = fuc(y - delta * A.T @ (A@y - b),delta*lam, alpha)
        v = fuc(x - delta * A.T @ (A@x - b),delta*lam, alpha)
     
        t_old = t
        t = (np.sqrt(4*t**2+1) + 1)/2
        
        x_old = x
        if obj(z) <= obj(v):
            x = z
        else:
            x = v
        relerr = norm(x-x_old)/max(norm(x),norm(x_old),eps)
        err = norm(x-x_g)/norm(x_g)
        relall.append(err)
        if relerr<reltol and i>2:
            break
    end = time.time()
    cost = end -start
    return cost,relall

def main():
    A,b,x_g = construct(64,256,1)
    cost,err = L1L2_ADMM(A,b,x_g,lam=1e-2)
    cost2,err2 = L1L2_DCA(A,b,x_g,lam=1e-2)
    cost3,err3 = L1L2_FBS(A,b,x_g,lam=1e-2)
    print("ADMM time cost:",cost)
    print("DCA time cost:",cost2)
    print("FBS time cost:",cost3)
    plt.figure(num=3,figsize=(8,5))
    plt.plot(np.log10(err))
    plt.plot(np.log10(err2),color='black',linewidth=1,linestyle=':')
    plt.plot(np.log10(err3),color='red',linewidth=1,linestyle='--')
    plt.legend(['ADMM','DCA','FBS'])
    plt.show()
    

if __name__ == '__main__':
    #python中*是点乘,矩阵乘法需要使用函数或者@
    main()