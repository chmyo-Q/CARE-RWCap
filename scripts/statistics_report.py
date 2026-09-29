"""Matched initial-seed blocks on a fixed benchmark, not common random paths."""
import numpy as np


def block_contrast(values, draws=20000, seed=20260919, stream_index=0):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite repeat-block differences')
    interval=None
    if len(x)>1:
        rng=np.random.default_rng(seed)
        # Preserve the archived first three comparison streams explicitly;
        # adding another contrast must not shift existing interval endpoints.
        for _ in range(stream_index):rng.integers(0,len(x),(draws,len(x)))
        means=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
        interval=np.quantile(means,[.025,.975]).tolist()
    return dict(mean_difference_pp=float(x.mean()),seed_block_differences_pp=x.tolist(),
        ci95_pp=interval,bootstrap_draws=draws if interval else 0,bootstrap_seed=seed,bootstrap_stream_index=stream_index,
        interpretation='Candidate minus control; negative favors candidate. Pointwise percentile interval over initial-seed blocks on this fixed benchmark; no multiple-comparison correction, no population generalization or common-trajectory assumption.')


def attribution(rows, plan):
    seeds=plan['seeds'];cases=plan['cases'];arms=plan['arms']
    table={(r['case'],r['initial_seed'],r['arm']):r for r in rows}
    values={}
    for arm in arms:
        for endpoint,key in [('raw','raw_self_error_percent'),('cer','s24_self_error_percent')]:
            values[arm,endpoint]=np.array([np.mean([table[c,s,arm][key] for c in cases]) for s in seeds])
    comparisons={}
    settings=plan.get('bootstrap',{})
    def add(name,x):
        streams={'whole method: bpr CER - p0 raw':0,'whole method: full CER - p0 raw':1,'cer: full - bpr':2}
        comparisons[name]=block_contrast(x,settings.get('draws',20000),settings.get('seed',20260919),streams.get(name,3))
    for arm in arms:
        add(arm+': CER - raw',values[arm,'cer']-values[arm,'raw'])
    for endpoint in ['raw','cer']:
        for control,candidate in [('p0','bpr'),('bpr','full')]:
            if control in arms and candidate in arms:
                add(f'{endpoint}: {candidate} - {control}',values[candidate,endpoint]-values[control,endpoint])
    if 'p0' in arms and 'full' in arms:
        add('whole method: full CER - p0 raw',values['full','cer']-values['p0','raw'])
    if 'p0' in arms and 'bpr' in arms:
        add('whole method: bpr CER - p0 raw',values['bpr','cer']-values['p0','raw'])
    return comparisons


def workload(rows):
    """Measured solver time, not wrapper/load time or a speedup assertion."""
    required=['walks','hops_per_walk','total_steps_approx','elapsed_seconds','cpu_seconds']
    if not all(all(k in r for k in required) for r in rows):
        return None
    cases=sorted({r['case'] for r in rows});walks=sum(r['walks'] for r in rows)
    return dict(mean_elapsed_seconds_per_case=float(np.mean([r['elapsed_seconds'] for r in rows])),
        mean_cpu_seconds_per_case=float(np.mean([r['cpu_seconds'] for r in rows])),
        mean_total_walks_across_cases=float(sum(np.mean([r['walks'] for r in rows if r['case']==c]) for c in cases)),
        walk_weighted_hops_per_walk=sum(r['walks']*r['hops_per_walk'] for r in rows)/walks if walks else None,
        mean_total_steps_approx_across_cases=float(sum(np.mean([r['total_steps_approx'] for r in rows if r['case']==c]) for c in cases)),
        peak_gpu_memory=None,peak_host_memory=None,
        note='Memory is not measured by this runner. Step counts use rounded solver hops. Timings describe this batch only; no automatic acceleration claim.')
