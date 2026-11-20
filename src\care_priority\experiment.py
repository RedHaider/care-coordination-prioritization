import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score

def dataset(n=2500,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,8));group=r.binomial(1,.35,n);logit=.9*x[:,0]+.7*x[:,1]-.5*x[:,2]+.8*np.maximum(x[:,3],0)*x[:,4]+.25*group-1.6;p=1/(1+np.exp(-logit));return x,group,r.binomial(1,p)
def evaluate(seed=27):
 x,g,y=dataset(seed=seed);cut=1700;m=HistGradientBoostingClassifier(max_depth=3,random_state=seed).fit(x[:cut],y[:cut]);p=m.predict_proba(x[cut:])[:,1];yt=y[cut:];gt=g[cut:];k=max(1,int(.1*len(p)));chosen=np.argsort(p)[-k:];positive=max(1,yt.sum());return {"auroc":float(roc_auc_score(yt,p)),"average_precision":float(average_precision_score(yt,p)),"brier_score":float(brier_score_loss(yt,p)),"recall_at_10pct_capacity":float(yt[chosen].sum()/positive),"group_1_share_selected":float(gt[chosen].mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
