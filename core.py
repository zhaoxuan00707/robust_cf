"""Wachter Base and update-aware GRACE, extracted from the verified Task100 code."""
from collections import OrderedDict
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from torch.func import functional_call
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ot

@dataclass
class ExactPlan:
    source_index: torch.Tensor
    target_index: torch.Tensor
    weights: torch.Tensor
    distance: float
    marginal_residual: float
    solver: str

    def differentiable_distance(self, source, target):
        i=self.source_index.to(source.device);j=self.target_index.to(target.device)
        weights=self.weights.to(source)
        return torch.linalg.vector_norm((source[i]-target[j])*weights.sqrt()[:,None])

    def state_dict(self):return dict(vars(self))

def exact_w2(source, target, a=None, b=None):
    """Balanced quadratic OT, with an exact equal-uniform assignment shortcut."""
    x=source.detach().cpu().double().numpy();y=target.detach().cpu().double().numpy()
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite compatible support matrices required')
    a=np.full(len(x),1/len(x)) if a is None else np.asarray(a,dtype=np.float64)
    b=np.full(len(y),1/len(y)) if b is None else np.asarray(b,dtype=np.float64)
    if (a<=0).any() or (b<=0).any() or abs(a.sum()-1)>1e-8 or abs(b.sum()-1)>1e-8:
        raise ValueError('positive probability masses required')
    cost=cdist(x,y,'sqeuclidean')
    if len(x)==len(y) and np.allclose(a,1/len(x),atol=1e-14,rtol=0) and np.allclose(b,1/len(y),atol=1e-14,rtol=0):
        i,j=linear_sum_assignment(cost);mass=a.copy();solver='SciPy linear_sum_assignment; exact equal-mass discrete quadratic OT'
    else:
        plan,log=ot.emd(a,b,cost,numItermax=1000000,log=True,numThreads=1)
        if log['warning'] is not None:raise RuntimeError(log['warning'])
        i,j=np.nonzero(plan>0);mass=plan[i,j];solver='POT network simplex; unregularized quadratic OT'
    residual=max(np.abs(np.bincount(i,weights=mass,minlength=len(x))-a).max(),np.abs(np.bincount(j,weights=mass,minlength=len(y))-b).max())
    if residual>1e-8:raise RuntimeError('OT marginals failed')
    distance=float(np.sqrt(np.sum(mass*cost[i,j])))
    return ExactPlan(torch.from_numpy(i.copy()),torch.from_numpy(j.copy()),torch.from_numpy(mass.copy()),distance,float(residual),solver)

def unroll(model, x, y, steps=3, lr=.01, differentiable=True):
    params=OrderedDict((k,v.detach().clone().requires_grad_(True)) for k,v in model.named_parameters())
    buffers=OrderedDict((k,v.detach().clone()) for k,v in model.named_buffers())
    for _ in range(steps):
        logits=functional_call(model,(params,buffers),(x,))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=differentiable)
        params=OrderedDict((k,p-lr*g) for (k,p),g in zip(params.items(),grads))
        if not differentiable:params=OrderedDict((k,p.detach().requires_grad_(True)) for k,p in params.items())
    return params,buffers

def differentiable_w2(a,b):
    plan=exact_w2(a.detach().flatten(1).cpu(),b.detach().flatten(1).cpu())
    i=plan.source_index.to(a.device);j=plan.target_index.to(b.device);w=plan.weights.to(a.device,dtype=a.dtype)
    squared=((a[i]-b[j]).flatten(1).square().sum(1)*w).sum()
    # At identical supports use the zero subgradient of the norm.
    distance=squared.sqrt() if float(squared.detach())>1e-20 else squared*0
    return distance,plan

class MixedDomain:
    def __init__(self, train, groups):
        self.groups = {k: list(v) for k, v in groups.items() if v}
        self.low = torch.zeros(train.shape[1], dtype=train.dtype, device=train.device)
        self.scale = torch.ones_like(self.low)
        for ids in self.groups.values():
            self.low[ids] = train[:, ids].amin(0)
            self.scale[ids] = train[:, ids].amax(0) - self.low[ids]
            if (self.scale[ids] <= 0).any():
                raise ValueError('Degenerate categorical coordinate')
        self.numeric = [j for j in range(train.shape[1]) if all(j not in v for v in self.groups.values())]

    def perturb(self, x, u, temperature=1., smoothing=1e-4):
        z = x + u
        low, scale = self.low.to(x), self.scale.to(x)
        for ids in self.groups.values():
            p = ((x[:, ids]-low[ids])/scale[ids]).clamp(0, 1)
            p = (1-smoothing)*p + smoothing/len(ids)
            p = p/p.sum(1, keepdim=True)
            alpha = temperature*p.log()
            z = z.clone()
            z[:, ids] = torch.softmax((alpha+u[:, ids])/temperature, 1)*scale[ids]+low[ids]
        return z

    def project(self, x, hard=False):
        z = x.clone(); low, scale = self.low.to(x), self.scale.to(x)
        for ids in self.groups.values():
            p = (x[:, ids]-low[ids])/scale[ids]
            if hard:
                p = torch.nn.functional.one_hot(p.argmax(1), len(ids)).to(x)
            else:
                # Euclidean projection in standardized coordinates: weighted simplex.
                weights = scale[ids].square()
                lo = ((p-1)*weights).amin(1, keepdim=True)
                hi = (p*weights).amax(1, keepdim=True)
                for _ in range(45):
                    mid = (lo+hi)/2
                    large = (p-mid/weights).clamp_min(0).sum(1, keepdim=True)>1
                    lo = torch.where(large, mid, lo); hi = torch.where(large, hi, mid)
                p = (p-(lo+hi)/2/weights).clamp_min(0)
            z[:, ids] = p*scale[ids]+low[ids]
        return z

def mixed_surrogate(model, x, y, q, r, domain, steps=10, lr=.01,
                    temperature=.05, cat_temperature=1., smoothing=1e-4):
    u = torch.zeros_like(x, requires_grad=True)
    perturbed = domain.perturb(x, u, cat_temperature, smoothing)
    params, buffers = unroll(model, perturbed, y, steps, lr, True)
    probability = functional_call(model, (params, buffers), (q,)).sigmoid()
    nominal = (temperature*torch.nn.functional.softplus((.5-probability)/temperature)).mean()
    derivative = torch.autograd.grad(nominal, u, create_graph=True)[0]
    sensitivity = math.sqrt(len(x))*torch.linalg.vector_norm(derivative)
    return nominal+r*sensitivity, nominal, sensitivity

def prototypes(train_x, groups):
    return {name:torch.unique(train_x[:,ids],dim=0) for name,ids in groups.items()}

def project(x, groups, protos):
    z=x.clone()
    for name,ids in groups.items():
        p=protos[name]
        distance=z[:,ids].square().sum(1,keepdim=True)+p.square().sum(1).unsqueeze(0)-2*z[:,ids]@p.T
        z[:,ids]=p[distance.argmin(1)]
    return z

def straight_through_project(x,groups,protos):
    px=project(x,groups,protos)
    return x+(px-x).detach()

def solve_wachter(model,source,threshold,groups,protos,restarts=5,steps=500,seed=0):
    records=[];batch=[]
    for restart in range(restarts):
        gen=torch.Generator().manual_seed(seed+restart);jitter=torch.zeros_like(source) if restart==0 else .02*torch.randn(source.shape,generator=gen);batch.append(source+jitter)
    expanded_source=source.repeat(restarts,1);x=torch.cat(batch).requires_grad_(True);opt=torch.optim.Adam([x],lr=.03);best=torch.full((len(x),),float("inf"));best_x=project(expanded_source,groups,protos);best_step=torch.full((len(x),),-1,dtype=torch.long)
    for step in range(steps):
        opt.zero_grad();xp=straight_through_project(x,groups,protos);score=torch.sigmoid(model(xp));loss=(20*torch.relu(threshold-score)+.05*(xp-expanded_source).square().mean(1)).mean();loss.backward();opt.step()
        with torch.no_grad():
            xp=project(x,groups,protos);score=torch.sigmoid(model(xp));cost=torch.linalg.vector_norm(xp-expanded_source,dim=1);use=(score>=threshold)&(cost<best);best[use]=cost[use];best_x[use]=xp[use];best_step[use]=step+1
    with torch.no_grad():score=torch.sigmoid(model(best_x));ok=score>=threshold
    solutions=[]
    for restart in range(restarts):
        sl=slice(restart*len(source),(restart+1)*len(source));solutions.append((best_x[sl].clone(),best[sl].clone(),ok[sl].clone(),best_step[sl].clone(),score[sl].clone()))
        for i in range(len(source)):records.append({"source_index":i,"confidence":threshold,"current_valid":bool(ok[sl][i]),"l2_cost":float(best[sl][i]) if ok[sl][i] else np.nan,"optimization_steps":int(best_step[sl][i]),"restart_id":restart,"final_classifier_score":float(score[sl][i]),"selected":False})
    # Per source, keep minimum valid restart, otherwise highest score.
    chosen=[]
    for i in range(len(source)):
        candidates=[(float(sol[1][i]),r) for r,sol in enumerate(solutions) if bool(sol[2][i])]
        r=min(candidates)[1] if candidates else max(range(restarts),key=lambda j:float(solutions[j][4][i]))
        chosen.append(solutions[r][0][i]);records[r*len(source)+i]["selected"]=True
    return torch.stack(chosen),records

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def score(model,z,futures,source):
    device=next(model.parameters()).device; q=z.to(device)
    with torch.no_grad():
        cv=(model(q).sigmoid()>=.5).cpu()
        predictions=torch.stack([(functional_call(model,(p,b),(q,)).sigmoid()>=.5).cpu() for p,b in futures])
    return dict(current_validity=float(cv.float().mean()),FV=float(predictions.float().mean()),
                Cost=float(torch.linalg.vector_norm(z-source,dim=1).mean()),predictions=predictions,current=cv)

def future_bank(model,x,y,domain,bref,cfg,offset):
    # Numerical shift families retain the old mean/scale definitions. Tail
    # resampling now moves labels with rows (old helper mismatched labels).
    numeric=domain.numeric
    corr=[]
    for j in numeric:
        v=np.corrcoef(x[:,j].cpu().numpy(),y.cpu().numpy())[0,1]
        corr.append(0. if not np.isfinite(v) else abs(float(v)))
    ids=[numeric[j] for j in np.argsort(corr)[-3:][::-1]]
    bank=[]; records=[]; states=[]
    for fi,family in enumerate(cfg['future']['families']):
      for si,severity in enumerate(cfg['future']['severities']):
       for k in range(cfg['future']['realizations']):
        seed=offset+fi*10000+si*100+k; gen=torch.Generator().manual_seed(seed)
        z=x.cpu().clone(); labels=y.cpu().clone(); target=severity*bref
        if family=='mean': z[:,ids]+=target/math.sqrt(len(ids))*torch.where(torch.rand(len(ids),generator=gen)>.5,1.,-1.)
        elif family=='scale':
            v=z[:,ids]; center=v.mean(0); norm=v.sub(center).square().sum(1).mean().sqrt().clamp_min(1e-6)
            z[:,ids]=center+(1+target/norm)*(v-center)
        else:
            v=z[:,ids[0]]; cut=v.median(); lo=(v<=cut).nonzero().flatten(); hi=(v>cut).nonzero().flatten()
            if not len(hi): hi=lo
            nlo=int(len(z)*min(.5+.18*severity,.95))
            index=torch.cat([lo[torch.randint(len(lo),(nlo,),generator=gen)],hi[torch.randint(len(hi),(len(z)-nlo,),generator=gen)]])
            z=z[index]; labels=labels[index]
        params,buffers=unroll(model,z.to(x),labels.to(y),**cfg['update'],differentiable=False)
        params={a:b.detach() for a,b in params.items()}; bank.append((params,buffers))
        states.append(({a:b.cpu() for a,b in params.items()},{a:b.cpu() for a,b in buffers.items()}))
        records.append(dict(family=family,severity=severity,realization=k,seed=seed,numeric_features=ids))
    return bank,states,records

def grace(model,x,y,base,p1,domain,r,beta,cfg):
    device=x.device; b=base.to(device); p1=p1.to(device); q=b.clone().requires_grad_(True)
    sync(); start=time.perf_counter()
    D=exact_w2(base,p1.cpu()).distance; budget=beta*D; dual=0.; previous=None; history=[]
    for step in range(cfg['solver']['steps']):
        f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'])
        g=differentiable_w2(b,q)[0]+differentiable_w2(q,p1)[0]-budget
        objective=float(f.detach()); change=math.inf if previous is None else abs(objective-previous)/max(1.,abs(previous))
        history.append(dict(step=step,objective=objective,nominal=float(nom.detach()),sensitivity=float(sens.detach()),violation=float(g.detach()),dual=dual,relative_change=change))
        if float(g.detach())<=cfg['solver']['feas_tol']*max(1.,D) and change<=cfg['solver']['obj_tol']: break
        loss=f+dual*g+.5*cfg['solver']['rho']*torch.relu(g).square()
        gradient=torch.autograd.grad(loss,q)[0]
        with torch.no_grad(): qnew=domain.project(q-cfg['solver']['primal_lr']*gradient)
        if not torch.isfinite(qnew).all(): raise RuntimeError('Nonfinite GRACE supports')
        q=qnew.detach().requires_grad_(True)
        newg=exact_w2(base,q.detach().cpu()).distance+exact_w2(q.detach().cpu(),p1.cpu()).distance-budget
        dual=max(0.,dual+cfg['solver']['rho']*newg)
        previous=objective
    z=q.detach().cpu(); plan=exact_w2(base,z); assignment=plan.target_index[plan.source_index.argsort()]; assigned=z[assignment]
    w1=plan.distance; w2=exact_w2(z,p1.cpu()).distance
    sync(); seconds=time.perf_counter()-start
    # Final diagnostics separately from generation runtime.
    f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'])
    return dict(q=z,cf=assigned,assignment=assignment,history=history,seconds=seconds,r=r,beta=beta,D=D,budget=budget,W_base_Q=w1,W_Q_P1=w2,feasible=w1+w2<=budget+cfg['solver']['feas_tol']*max(1.,D),nominal=float(nom.detach()),sensitivity=float(sens.detach()),surrogate=float(f.detach()),iterations=len(history))

