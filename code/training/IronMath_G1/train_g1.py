from g1_utils import *
import torch
from data import P,tensor,labels

def train_one(mid,arch,length,seed):
    registrations_ok();assert read('QUALITY_PRETRAIN.json')['passed']
    marker=R/'assets'/f'{mid}.json'
    if marker.exists():return read(f'assets/{mid}.json')
    folder=R/'checkpoints'/mid;folder.mkdir(parents=True,exist_ok=True)
    if list(folder.glob('*.pt')):raise RuntimeError('Unfinished model checkpoints present; no automatic retraining')
    torch.manual_seed(seed);model=make(arch)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    rng=np.random.default_rng(10000+seed);vx=np.random.default_rng(20000+seed).choice(4,(512,length),p=P).astype(np.uint8)
    vy=torch.from_numpy(labels(vx).astype(np.int64));vm=torch.from_numpy(vx==3);vt=tensor(vx)
    history=[];best_loss=float('inf');best_update=0;t=time.perf_counter();c=time.process_time();count=correct=0;loss_sum=0.
    dump(f'configs/{mid}.json',dict(model=mid,architecture=arch,H=64,state_size=model.state_size,L=length,seed=seed,train_seed=10000+seed,validation_seed=20000+seed,parameters=sum(p.numel() for p in model.parameters()),training=read('PREREGISTRATION.json')['training']))
    for update in range(1,1501):
        xx=rng.choice(4,(32,length),p=P).astype(np.uint8);yy=torch.from_numpy(labels(xx).astype(np.int64));mask=torch.from_numpy(xx==3)
        n=int(mask.sum());assert n>0
        opt.zero_grad(set_to_none=True);z=model(tensor(xx));loss=torch.nn.functional.cross_entropy(z[mask],yy[mask]);assert torch.isfinite(loss),'nonfinite training loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        count+=n;correct+=int((z.detach().argmax(-1)[mask]==yy[mask]).sum());loss_sum+=float(loss.detach())*n
        if update%100==0:
            model.eval()
            with torch.inference_mode():
                zv=model(vt);vl=float(torch.nn.functional.cross_entropy(zv[vm],vy[vm]));vc=int((zv.argmax(-1)[vm]==vy[vm]).sum())
            assert math.isfinite(vl)
            if vl<best_loss:best_loss=vl;best_update=update
            history.append(dict(update=update,train_loss=loss_sum/count,train_R_correct=correct,train_R_count=count,validation_loss=vl,validation_R_correct=vc,validation_R_count=int(vm.sum())))
            cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),architecture=arch,H=64,L=length,seed=seed,update=update,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history,best_loss=best_loss,best_update=best_update,wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-c,parameters=sum(p.numel() for p in model.parameters()))
            torch.save(cp,folder/f'u{update:04}.pt');model.train();count=correct=0;loss_sum=0.
    path=folder/f'u{best_update:04}.pt';csvout(f'history/{mid}.csv',history)
    asset=dict(model=mid,architecture=arch,H=64,L=length,seed=seed,state_size=model.state_size,path=path.relative_to(R).as_posix(),SHA256=sha(path),selected_update=best_update,validation_loss=best_loss,final_path=(folder/'u1500.pt').relative_to(R).as_posix(),parameters=cp['parameters'],train_wall_seconds=cp['wall_seconds'],train_CPU_seconds=cp['CPU_seconds'],peak_RAM_bytes=memory(),updates=1500,presented_sequences=48000,presented_positions=48000*length)
    dump(f'assets/{mid}.json',asset);print('TRAINED',mid,'selected',best_update,'wall',round(asset['train_wall_seconds'],2),flush=True)
    return asset
