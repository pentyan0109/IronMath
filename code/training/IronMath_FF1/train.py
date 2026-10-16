import time,copy
from common import *
def train_all():
    import numpy as np,torch
    from model import FF
    from data import P,tensor,labels
    registrations_ok();assert read('QUALITY_PRETRAIN.json')['passed'];assert not read('WORKSPACE_BEFORE.json')['errors'];process_gate()
    if not (R/'TRAINING_STARTED.json').exists():dump('TRAINING_STARTED.json',dict(utc=now(),registration_SHA256=sha(R/'PREREGISTRATION.json')))
    assets={};resources=[]
    for mid,h,l,seed in model_ids():
        folder=R/'checkpoints'/mid;folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
        model=FF(h);optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        rng=np.random.default_rng(10000+seed);vx=np.random.default_rng(20000+seed).choice(4,(512,l),p=P).astype(np.uint8);vy=torch.from_numpy(labels(vx).astype(np.int64));vm=torch.from_numpy(vx==3);xt=tensor(vx)
        history=[];best_loss=float('inf');best_update=0;start_update=0;oldwall=oldcpu=0.
        paths=sorted(folder.glob('u*.pt'))
        if paths:
            cp=torch.load(paths[-1],map_location='cpu',weights_only=False);assert (cp['H'],cp['L'],cp['seed'])==(h,l,seed)
            model.load_state_dict(cp['model']);optim.load_state_dict(cp['optimizer']);rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng']);history=cp['history'];best_loss=cp['best_loss'];best_update=cp['best_update'];start_update=cp['update'];oldwall=cp['wall_seconds'];oldcpu=cp['CPU_seconds']
        t=time.perf_counter();cpu=time.process_time();loss_sum=count=correct=0;model.train()
        for u in range(start_update+1,1501):
            xx=rng.choice(4,(32,l),p=P).astype(np.uint8);yy=torch.from_numpy(labels(xx).astype(np.int64));mask=torch.from_numpy(xx==3);optim.zero_grad(set_to_none=True);z=model(tensor(xx));n=int(mask.sum())
            if not n:raise RuntimeError('unexpected R-free batch: protocol clarification needed')
            loss=torch.nn.functional.cross_entropy(z[mask],yy[mask]);assert torch.isfinite(loss),'nonfinite training loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optim.step();loss_sum+=float(loss.detach())*n;count+=n;correct+=int((z.detach().argmax(-1)[mask]==yy[mask]).sum())
            if u%100==0:
                model.eval()
                with torch.inference_mode():
                    zv=model(xt);vl=float(torch.nn.functional.cross_entropy(zv[vm],vy[vm]));va=int((zv.argmax(-1)[vm]==vy[vm]).sum())
                assert torch.isfinite(torch.tensor(vl))
                if vl<best_loss:best_loss=vl;best_update=u
                history.append(dict(update=u,train_loss=loss_sum/count,train_R_correct=correct,train_R_count=count,validation_loss=vl,validation_R_correct=va,validation_R_count=int(vm.sum())))
                cp=dict(model=model.state_dict(),optimizer=optim.state_dict(),H=h,L=l,seed=seed,update=u,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history,best_loss=best_loss,best_update=best_update,wall_seconds=oldwall+time.perf_counter()-t,CPU_seconds=oldcpu+time.process_time()-cpu,parameters=sum(p.numel() for p in model.parameters()))
                path=folder/f'u{u:04}.pt';assert not path.exists();torch.save(cp,path);loss_sum=count=correct=0;model.train()
        selected=folder/f'u{best_update:04}.pt';endcp=torch.load(folder/'u1500.pt',map_location='cpu',weights_only=False)
        assets[mid]=dict(path=selected.relative_to(R).as_posix(),SHA256=sha(selected),H=h,L=l,seed=seed,selected_update=best_update,validation_loss=best_loss,parameters=sum(p.numel() for p in model.parameters()),final_path=(folder/'u1500.pt').relative_to(R).as_posix())
        resources.append(dict(model=mid,H=h,L=l,seed=seed,wall_seconds=endcp['wall_seconds'],CPU_seconds=endcp['CPU_seconds'],peak_RAM_bytes=memory(),updates=1500,presented_sequences=48000,presented_positions=48000*l,selected_update=best_update,parameters=assets[mid]['parameters']))
        if not (R/f'history/{mid}.csv').exists():csvout(f'history/{mid}.csv',history)
        print('TRAINED',mid,'selected',best_update,'wall',round(endcp['wall_seconds'],2),flush=True)
    dump('CHECKPOINTS_FROZEN.json',dict(utc=now(),models=assets,training_models=40,new_conditions=0));csvout('training_resources.csv',resources)
    dump('TRAINING_COMPLETE.json',dict(utc=now(),models=40,updates_per_model=1500,checkpoint_manifest_SHA256=sha(R/'CHECKPOINTS_FROZEN.json')))
