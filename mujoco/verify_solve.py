import numpy as np
# structured-convex contact solve (normal contacts, learned regularization R=softplus(eta))
#   lambda* = argmin 1/2 l'(W+diag(R))l + q'l  s.t. l>=0     (strictly convex, unique)
# verify implicit gradient dL/deta (L=1/2||lambda*||^2) vs finite differences.
sig = lambda x: 1/(1+np.exp(-x))
sp  = lambda x: np.log1p(np.exp(-np.abs(x)))+np.maximum(x,0)

def solve_nn_qp(A,q):
    n=len(q); best=None; bv=np.inf
    for mask in range(1<<n):
        I=[i for i in range(n) if (mask>>i)&1]; x=np.zeros(n)
        if I:
            xi=np.linalg.solve(A[np.ix_(I,I)],-q[I])
            if np.any(xi<-1e-12): continue
            x[I]=xi
        g=A@x+q; Aset=[i for i in range(n) if not((mask>>i)&1)]
        if Aset and np.any(g[Aset]<-1e-9): continue
        val=0.5*x@A@x+q@x
        if val<bv: bv,best=val,(x,I)
    return best

def forward(eta,W,q):
    R=sp(eta); x,I=solve_nn_qp(W+np.diag(R),q); return x,I,R

def grad_analytic(eta,W,q):
    x,I,R=forward(eta,W,q); g=np.zeros(len(q))
    if I:
        idx=np.array(I); Aii=(W+np.diag(R))[np.ix_(idx,idx)]; Ainv=np.linalg.inv(Aii)
        for k in I:
            pk=list(idx).index(k); E=np.zeros_like(Aii); E[pk,pk]=sig(eta[k])
            dxI=-Ainv@(E@x[idx]); g[k]=x[idx]@dxI
    return g,x

np.random.seed(1); n=4
B=np.random.randn(n,n); W=B@B.T+0.1*np.eye(n); q=np.random.randn(n)-0.6; eta=np.random.randn(n)
ga,x=grad_analytic(eta,W,q)
gf=np.zeros(n); eps=1e-6
for k in range(n):
    e=eta.copy(); e[k]+=eps; xp,_,_=forward(e,W,q)
    e=eta.copy(); e[k]-=eps; xm,_,_=forward(e,W,q)
    gf[k]=(0.5*xp@xp-0.5*xm@xm)/(2*eps)
print("lambda* =",np.round(x,4)," (active set:",[i for i in range(n) if x[i]>1e-9],")")
print("grad implicit =",np.round(ga,6))
print("grad finite   =",np.round(gf,6))
print("max abs err   = %.2e"%np.abs(ga-gf).max())
