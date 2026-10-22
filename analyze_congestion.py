#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from validate_congestion import FEATURE_NAMES, make, metrics

TRAIN=list(range(1000,1050)); TEST=list(range(2000,2050))
Xtr,ytr=make(TRAIN); Xte,yte=make(TEST)
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-9]=1
A=(Xtr-mu)/sd; B=(Xte-mu)/sd

def ridge_fit(x,y,lam=1.0):
    z=np.c_[np.ones(len(x)),x]
    w=np.linalg.solve(z.T@z+lam*np.eye(z.shape[1]),z.T@y)
    return lambda q: np.c_[np.ones(len(q)),q]@w

def relu_fit(x,y,hidden=8,epochs=1000,lr=.01,seed=7,loss='mse'):
    r=np.random.default_rng(seed); w1=r.normal(0,.08,(x.shape[1],hidden)); b1=np.zeros(hidden); w2=r.normal(0,.08,(hidden,1)); b2=np.array([y.mean()])
    for _ in range(epochs):
        z=x@w1+b1; h=np.maximum(z,0); p=(h@w2+b2).ravel(); e=p-y
        if loss=='huber':
            g=np.where(np.abs(e)<=.1,e,.1*np.sign(e))
        elif loss=='log':
            # trained on log1p targets outside this function
            g=e
        else: g=e
        d=(2/len(y))*g[:,None]; dw2=h.T@d; db2=d.sum(0); dz=(d@w2.T)*(z>0)
        w2-=lr*dw2; b2-=lr*db2; w1-=lr*x.T@dz; b1-=lr*dz.sum(0)
    return lambda q:(np.maximum(q@w1+b1,0)@w2+b2).ravel()

def evalp(name,p):
    m=metrics(yte,p); return {'model':name,**m}

# diagnostics
stats={'train_target_mean':float(ytr.mean()),'test_target_mean':float(yte.mean()),'train_zero_rate':float(np.mean(ytr==0)),'test_zero_rate':float(np.mean(yte==0)),'train_max':float(ytr.max()),'test_max':float(yte.max())}
cor=[]
for i,n in enumerate(FEATURE_NAMES):
    cor.append({'feature':n,'pearson':float(np.corrcoef(Xtr[:,i],ytr)[0,1]) if Xtr[:,i].std()>0 else 0.0,'spearman':float(np.corrcoef(np.argsort(np.argsort(Xtr[:,i])),np.argsort(np.argsort(ytr)))[0,1])})

results=[]
meanp=np.full_like(yte,ytr.mean()); results.append(evalp('mean_baseline',meanp))
for lam in [0.01,.1,1,10,100]: results.append(evalp(f'ridge_lambda_{lam}',ridge_fit(A,ytr,lam)(B)))
for hidden in [4,8,16,32]:
  for loss in ['mse','huber']:
    p=relu_fit(A,ytr,hidden=hidden,epochs=1500,lr=.01,seed=7,loss=loss)(B); results.append(evalp(f'relu_{hidden}_{loss}',p))
# log target model, convert back to original scale
lp=relu_fit(A,np.log1p(ytr),hidden=8,epochs=1500,lr=.01,seed=7,loss='log')(B); results.append(evalp('relu_8_log1p_backtransform',np.expm1(lp).clip(0)))

# permutation importance for best non-baseline by RMSE
best=min([r for r in results if r['model']!='mean_baseline'], key=lambda r:r['rmse'])['model']
if best.startswith('ridge'):
    lam=float(best.split('_')[-1]); pred_fn=lambda q:ridge_fit(A,ytr,lam)(q)
elif best.startswith('relu_'):
    parts=best.split('_'); h=int(parts[1]); loss=parts[2]; pred_fn=lambda q:relu_fit(A,ytr,h,1500,.01,7,loss)(q)
base_pred=pred_fn(B); base_rmse=metrics(yte,base_pred)['rmse']; perm=[]
rng=np.random.default_rng(123)
for i,n in enumerate(FEATURE_NAMES):
    vals=[]
    for _ in range(30):
        q=B.copy(); q[:,i]=rng.permutation(q[:,i]); vals.append(metrics(yte,pred_fn(q))['rmse'])
    perm.append({'feature':n,'rmse_increase':float(np.mean(vals)-base_rmse),'permuted_rmse':float(np.mean(vals))})
perm.sort(key=lambda x:x['rmse_increase'],reverse=True)
report={'data':stats,'correlations':sorted(cor,key=lambda x:abs(x['pearson']),reverse=True),'model_results':sorted(results,key=lambda x:x['rmse']),'permutation_importance_model':best,'permutation_importance':perm}
Path('benchmark_results/congestion_model_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
