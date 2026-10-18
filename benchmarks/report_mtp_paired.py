"""Audit paired timings and separate traces; export tables and publication plots."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics as stats


def mean(row, key):
    return stats.mean(row['metrics'][key])


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |',
                      '| '+' | '.join('---' for _ in header)+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def audit(timing, profiles, root):
    samples = timing['samples']
    expected_cases = {(1,64,128),(1,64,512),(1,1024,128),(1,1024,512),
                      (2,64,512),(4,64,128),(4,64,512),(8,64,128),(8,64,512)}
    if len(samples) != 90 or len(timing['engines']) != 90:
        raise ValueError('Expected 90 samples and independent engines')
    groups = defaultdict(list)
    for i, (row, engine) in enumerate(zip(samples, timing['engines']), 1):
        key = row['batch'], row['input'], row['output']
        groups[key].append(row)
        if not row['strict_pass'] or not row['actual_commits_pass'] or row['execution_order'] != i:
            raise ValueError('Strict completion/commit/execution-order gate')
        if any(engine[k] != row[k] for k in ('batch','pair','mode')):
            raise ValueError('Engine/sample alignment')
        if row['metrics']['tokens'] != (row['output']-1)*row['batch']:
            raise ValueError('Actual token denominator')
        if row['mode'] == 'off' and (engine['mtp_loaded'] or engine['mtp_cache_bytes'] or engine['snapshot_bytes']):
            raise ValueError('Off engine includes MTP resources')
    if set(groups) != expected_cases:
        raise ValueError('Incomplete purpose matrix')
    for key, rows in groups.items():
        if len(rows) != 10:
            raise ValueError('Exactly five complete pairs required')
        for pair in range(1,6):
            pair_rows = [r for r in rows if r['pair']==pair]
            modes = ['off','reuse'] if pair%2 else ['reuse','off']
            if [r['mode'] for r in pair_rows] != modes or pair_rows[1]['execution_order'] != pair_rows[0]['execution_order']+1:
                raise ValueError('Pair order/adjacency')
        if any(r['completion'] != rows[0]['completion'] for r in rows):
            raise ValueError('Within-case completion differs')
        b,n,o = key
        if o==128 and rows[0]['completion'] != [v[:128] for v in groups[(b,n,512)][0]['completion']]:
            raise ValueError('Output-length prefix consistency')
    if timing['source']['sha256'] != profiles['source_sha256']:
        raise ValueError('Timing/profile source mismatch')
    for path, digest in timing['source']['sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Production source changed')
    if set(profiles['profiles']) != {f'B{b}-{m}' for b in (2,4,8) for m in ('off','reuse')}:
        raise ValueError('Profile coverage')
    for key, value in profiles['profiles'].items():
        b = int(key.split('-')[0][1:])
        expected = groups[(b,64,512)][0]['completion']
        for phase in ('prefill_trace','decode_trace','host','events'):
            record = value[phase]
            if not record['strict_pass'] or record['completion'] != expected:
                raise ValueError('Profile full completion gate')
            if phase.endswith('_trace'):
                trace = Path(record['trace'])
                if not trace.is_absolute():
                    trace = root/trace
                if hashlib.sha256(trace.read_bytes()).hexdigest() != record['gzip_sha256']:
                    raise ValueError('Trace artifact changed')
    return groups


def render(timing, profiles, groups, out):
    rows = []
    csv_rows = []
    for key in sorted(groups):
        values = groups[key]
        b,n,o = key
        summary = timing['summaries'][f'B{b}-I{n}-O{o}']
        off = [r for r in values if r['mode']=='off']
        reuse = [r for r in values if r['mode']=='reuse']
        speed = summary['e2e_speedup_pct']
        ratio = summary['tpot_speedup_pct']
        rows.append([b,n,o,
                     f"{stats.mean(r['metrics']['e2e_s'] for r in off):.3f} / {stats.mean(r['metrics']['e2e_s'] for r in reuse):.3f}",
                     f"{speed['geometric_mean']:+.1f}% [{speed['ci95'][0]:+.1f}, {speed['ci95'][1]:+.1f}]",
                     f"{stats.mean(mean(r,'request_ttft_ms') for r in off):.1f} / {stats.mean(mean(r,'request_ttft_ms') for r in reuse):.1f}",
                     f"{stats.mean(mean(r,'request_tpot_ms') for r in off):.3f} / {stats.mean(mean(r,'request_tpot_ms') for r in reuse):.3f}",
                     f"{ratio['geometric_mean']:+.1f}% [{ratio['ci95'][0]:+.1f}, {ratio['ci95'][1]:+.1f}]",
                     f"{100*stats.mean(r['acceptance_rate'] for r in reuse):.1f}%"])
        for row in values:
            m=row['metrics']
            csv_rows.append(dict(case=row['case'],mode=row['mode'],pair=row['pair'],order=row['execution_order'],
                batch=b,input=n,output=o,e2e_s=m['e2e_s'],ttft_ms=mean(row,'request_ttft_ms'),
                tpot_ms=mean(row,'request_tpot_ms'),decode_tokens_per_s=m['tokens_per_s'],
                acceptance=row['acceptance_rate'],actual_decode_tokens=m['tokens'],
                peak_mib=m['peak_allocated_bytes']/2**20,strict_pass=row['strict_pass']))
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csv_rows[0]);w.writeheader();w.writerows(csv_rows)
    sections = ['# 4B FP8 MTP：独立 off engine、5次配对及新 trace',
        '## 正式性能',
        '每项均为5对，单元格按 off / reuse 展示。端到端和 TPOT 的加速为每对 off/reuse 的几何均值减1；正值代表 reuse 更快，**不是耗时下降百分比**。括号为5对 log-ratio 的描述性95% t 区间（df=4），不包含跨硬件、真实负载或多重比较保证。绝对耗时为5次算术均值。',
        table(['B','输入','输出','完整请求 s','端到端加速 [95%区间]','TTFT ms','TPOT ms','TPOT加速 [95%区间]','接受率'],rows),
        '![端到端配对收益](paired_e2e.png)',
        '![TPOT和吞吐](batch_decode.png)',
        '## 每对结果及波动',
        '顺序固定为奇数对 off→reuse、偶数对 reuse→off；保留全部样本。每对完整结果见 samples.csv，逐步提交数和全部completion见 results.json。']
    amortization=[]
    for key in sorted(groups):
        b,n,o=key
        rows=groups[key]
        delta=[]
        startup=[]
        decode=[]
        for pair in range(1,6):
            off=next(r for r in rows if r['pair']==pair and r['mode']=='off')
            reuse=next(r for r in rows if r['pair']==pair and r['mode']=='reuse')
            delta.append(1000*(off['metrics']['e2e_s']-reuse['metrics']['e2e_s']))
            startup.append(mean(reuse,'request_ttft_ms')-mean(off,'request_ttft_ms'))
            decode.append((o-1)*(mean(off,'request_tpot_ms')-mean(reuse,'request_tpot_ms')))
        amortization.append([b,n,o,f'{stats.mean(startup):+.2f}',f'{stats.mean(decode):+.2f}',f'{stats.mean(delta):+.2f}'])
    sections[6:6]=['## 启动与decode的配对耗时分解',
        '下表先在每对中做差再取算术均值。TTFT增加为负时表示reuse启动更快；decode节省及完整请求节省为正时表示reuse更快。固定同长同步完成请求下，完整请求节省约等于decode节省减TTFT增加，余量是计时边界开销。两个输出长度并非同一时刻测量，差异不应全部归因于摊销。',
        table(['B','输入','输出','TTFT增加ms','decode节省ms/请求','完整请求节省ms'],amortization)]
    for key in sorted(groups):
        b,n,o=key
        summary=timing['summaries'][f'B{b}-I{n}-O{o}']
        lines=[]
        for label,field in (('完整请求加速%','e2e_speedup_pct'),('TPOT加速%','tpot_speedup_pct'),('TTFT增加ms','ttft_extra_ms')):
            d=summary[field]
            lines.append([label,', '.join(f'{v:+.2f}' for v in d['values']),
                          f"{d['mean']:+.2f}",f"{d['median']:+.2f}",f"{d['stdev']:.2f}"])
        sections += [f'### B{b} / I{n} / O{o}',table(['口径','第1～5对','算术均值','中位数','样本标准差'],lines)]
    graph_rows=[]
    for key in sorted(groups):
        for mode in ('off','reuse'):
            buckets=defaultdict(lambda:dict(graph=0,eager=0,batches=defaultdict(int)))
            for row in groups[key]:
                if row['mode']!=mode:continue
                for stage,v in row['graphs'].items():
                    dest=buckets[stage];dest['graph']+=v['graph_calls'];dest['eager']+=v['eager_calls']
                    for b,count in v['batch_counts'].items():dest['batches'][b]+=count
            for stage,d in buckets.items():
                total=d['graph']+d['eager']
                graph_rows.append(['/'.join(map(str,key)),mode,stage,d['graph'],d['eager'],f"{100*d['graph']/total:.1f}%",dict(d['batches'])])
    sections += ['## Graph命中与活跃shape',
        '以下是5次正式运行之和，按dispatch调用计数。prefill的Graph指生产piecewise路径，不能等同于整个prefill均被捕获。固定同长请求的本轮负载不代表持续到达或动态batch命中率。',
        table(['B/I/O','模式','阶段','Graph调用','eager调用','调用命中率','实际batch:次数'],graph_rows)]
    projection_rows=[]
    for key in sorted(groups):
        rows=[r for r in groups[key] if r['mode']=='reuse']
        values=[]
        for row in rows:
            g=row['graphs']
            def trials(stage):
                return sum(int(b)*count for b,count in g.get(stage,{}).get('batch_counts',{}).items())
            commits=row['metrics']['tokens']
            target=2*trials('verify')+trials('decode')
            draft=trials('draft')
            values.append((target/commits,draft/commits,(target+draft)/commits,trials('cache_only')/commits))
        projection_rows.append(['/'.join(map(str,key)),*[f'{stats.mean(v[i] for v in values):.3f}' for i in range(4)]])
    sections += ['## 接受率之外：每个实际提交token需要计算多少行',
        '由正式dispatch计数计算，排除首轮prefill。普通decode每个提交token对应1行target/head。reuse每轮验证2B行target/head，另外为后续draft执行相应行head；target列也对应GDN顺序更新行数。为保持普通decode数值语义，compute_verify_logits仍执行两次M=B词表投影，draft再执行一次相应batch投影；它们尚未像主干dense一样合并权重读取。下表是计算行数，不能直接当作耗时比例；dense权重读取可共享，但多余输出行和MTP词表投影并没有消失。',
        table(['B/I/O','target/head行/token','draft head行/token','head总行/token','cache-only行/token'],projection_rows)]
    trace_rows=[]
    group_rows=[]
    category_rows=[]
    for key,v in profiles['profiles'].items():
        for phase in ('prefill','decode'):
            d=v[phase+'_trace'];commits=d['actual_commits']
            counts=f"{min(d['rows'][0]['before']['completed'])}→{min(d['rows'][-1]['after']['completed'])}"
            request_rounds=sum(row['before']['active'] for row in d['rows'])
            per_round=commits/request_rounds
            trace_rows.append([key,phase,counts,len(d['rows']),commits,f'{per_round:.3f}',f"{d['kernel_sum_us']/1000:.2f}",
                f"{d['device_busy_union_us']/1000:.2f}",f"{d['device_span_us']/1000:.2f}",
                f"{100*d['device_internal_gap_us']/d['device_span_us']:.1f}%",
                f"{d['decision_wait_cpu_us']/1000:.2f}",f"{d['decision_wait_gpu_overlap_us']/1000:.2f}",
                f"{d['decision_wait_without_gpu_us']/1000:.2f}"])
            for group,us in sorted(d['kernel_groups_us'].items(),key=lambda x:-x[1]):
                group_rows.append([key,phase,group,f'{us/1000:.3f}',f'{100*us/d["kernel_sum_us"]:.1f}%',f'{us/commits:.1f}'])
            if phase=='decode':
                total=defaultdict(float)
                for cats in d['kernel_categories_us'].values():
                    for name,us in cats.items():total[name]+=us
                for name,us in sorted(total.items(),key=lambda x:-x[1]):
                    category_rows.append([key,name,f'{us/1000:.3f}',f'{100*us/d["kernel_sum_us"]:.1f}%',f'{us/commits:.1f}'])
    sections += ['## 独立诊断trace：prefill与稳态decode',
        '每种模式独立engine；prefill只截首轮，decode截completion计数112至144（reuse可能跨过末端1个token），以窗口实际提交token数归一化。trace后继续完成512token并严格比对正式off输出。Profiler运行不用于正式TPOT或端到端计时。',
        table(['配置','窗口','实际completion区间','engine轮数','实际提交','提交/请求/轮','kernel累计ms','设备忙碌并集ms','设备首尾跨度ms','跨度内空隙','决策等待CPUms','等待与GPU重合ms','等待无GPU重合ms'],trace_rows),
        '短窗口接受率不一定等于全512token的平均接受率。固定B且全为verify的decode窗口中，提交/请求/轮减1即窗口接受率；归因时同时参考完整事件pass，不能把短窗口收益外推到整个请求。',
        'GPU空隙为首个到末个physical device event之间的空白，不覆盖窗口两端；不能直接解释为全部可消除CPU开销。Profiler会放大细粒度kernel的调度开销，因此这里的空隙属于诊断pass，不能直接换算成正式TPOT改善。决策等待包含GPU执行等待，与GPU重合部分不能重复算成优化空间。',
        '### 互斥kernel归因',
        '按同trace GPU annotation最内层归因；draft graph中无法直接标注的head使用本trace target-head的kernel name/grid/block指纹，若不匹配则仍计入draft。应与下方独立CUDA事件head数据交叉判断，不能把draft_head缺失解释为没有词表投影。',
        table(['配置','窗口','组','kernel累计ms','累计占比','us/实际提交token'],group_rows),
        '### 稳态decode算子类别',table(['配置','类别','kernel累计ms','累计占比','us/实际提交token'],category_rows),
        '![稳态decode算子成本](trace_decode.png)']
    stage_rows=[]
    host_rows=[]
    for key,v in profiles['profiles'].items():
        for phase in ('prefill','decode'):
            commits=int(key.split('-')[0][1:]) if phase=='prefill' else v['events']['actual_decode_commits']
            rounds=1 if phase=='prefill' else len(v['events']['rows'])-1
            for label,ms in v['events'][phase].items():
                stage_rows.append([key,phase,label,f'{ms:.3f}',f'{ms/rounds:.3f}',f'{1000*ms/commits:.1f}'])
            for label,ms in v['host'][phase].items():
                host_rows.append([key,phase,label,f'{ms:.3f}'])
    sections += ['## 独立CUDA事件和host诊断',
        '事件pass需要重新capture并逐步同步；host pass只读CPU scope。两者均与正式计时、未重capture的物理trace分开。target/target_body/target_head、draft/draft_body/draft_head等存在父子嵌套，**不得相加当作总耗时或互斥占比**。事件时间包含该阶段stream间隙，kernel表只累计物理kernel，二者口径不同。',
        table(['配置','阶段','事件scope','累计GPU ms','ms/engine轮','us/实际提交token'],stage_rows),
        table(['配置','阶段','host scope','累计CPU ms'],host_rows),
        '## engine资源',
        '下表为每个case第一对的独立engine，单位MiB。相同GPU显存比例下，off释放MTP权重/状态后可能把更多预算分配给target KV；这属于真实off engine口径。初始化/加载/graph capture不计入请求TTFT与e2e。']
    engines=[]
    for key in sorted(groups):
        for mode in ('off','reuse'):
            row=next(r for r in groups[key] if r['pair']==1 and r['mode']==mode)
            e=timing['engines'][row['execution_order']-1]
            engines.append(['/'.join(map(str,key)),mode,e['prefill_graph'],
                *[f'{e[k]/2**20:.1f}' for k in ('mtp_weights_bytes','mtp_cache_bytes','snapshot_bytes','target_cache_bytes')]])
    sections += [table(['B/I/O','模式','prefill Graph','MTP权重','MTP KV','snapshot','target KV'],engines),
        '## 验证与适用范围',
        '- 90/90正式样本，9配置×5完整配对；交替且相邻顺序、独立off资源、严格completion、实际提交token分母、128/512前缀一致性均已审计。',
        '- 6个engine配置，各4个独立诊断pass；所有pass完成512token并严格一致；12个压缩trace校验SHA256。',
        '- 正式计时记录各batch预热后的完整Triton配置快照；每个样本前后均检查没有新增或替换选择。诊断在engine构造前恢复同一候选配置，已有选择不符会报错。',
        '- mtp2生产源码SHA256与正式计时及诊断完全一致。main通用改进独立提交，未将MTP协议移入main。本轮无MTP生产代码性能改动。',
        '- 本轮90样本的新门禁为完整completion、实际提交数、源码及调优不变；没有在每个正式样本中逐层转储GDN/KV。状态正确性沿用冻结基线已有回归，并在独立插桩回归中比较serial/reuse的hidden、logits及相同已消费前缀状态，不能仅凭completion一致宣称MTP cache逐位一致。',
        f"- 环境：{timing['environment']['gpu']}；Torch {timing['environment']['torch']}；CUDA {timing['environment']['cuda']}。target={timing['model']}；MTP={timing['mtp_model']}。",
        '- 固定同一synthetic prompt、greedy、ignore_eos、TP1、per-channel 4B。不能外推到block-wise、9B、5090、多温度或真实动态负载。5对仍可能受同机运行噪声影响，CI跨零时收益方向未得到清晰支持。',
        '- main回归：安全限制107通过；采样行118通过（含BF16/两种FP8、eager/Graph的hidden/输出/GDN/KV门禁）；调度123通过；Graph统计124通过。main工作区保持干净。',
        '- main提交：a91c95a（序列化/能力限制）、cfda31c（采样行投影）、02d9e18（decode优先/slot所有权）、4c43f71（Graph统计）。均为本地提交，尚未push。']
    (out/'measurements.md').write_text('\n\n'.join(sections)+'\n')


def plots(timing, profiles, groups, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    keys=sorted(groups)
    fig,ax=plt.subplots(figsize=(11,5))
    for i,(b,n,o) in enumerate(keys):
        d=timing['summaries'][f'B{b}-I{n}-O{o}']['e2e_speedup_pct']
        y=d['geometric_mean'];lo,hi=d['ci95']
        ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',color='#137b80',capsize=4)
        ax.scatter([i]*5,d['values'],s=14,color='#555555',alpha=.65)
    ax.axhline(0,color='#888',linewidth=1)
    ax.set_xticks(range(len(keys)),[f'B{b}\nI{n}/O{o}' for b,n,o in keys])
    ax.set_ylabel('Paired e2e speedup (%) = off / reuse - 1')
    ax.set_title('Five adjacent independent-engine pairs; dots = pairs, bars = descriptive 95% CI')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'paired_e2e.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode,color in (('off','#555555'),('reuse','#137b80')):
        for ax,field,label in ((axes[0],'tpot','TPOT (ms/token/request)'),(axes[1],'throughput','Aggregate decode throughput (token/s)')):
            y=[]
            for b in (1,2,4,8):
                rows=[r for r in groups[(b,64,512)] if r['mode']==mode]
                values=[mean(r,'request_tpot_ms') if field=='tpot' else r['metrics']['tokens_per_s'] for r in rows]
                y.append(stats.mean(values));ax.scatter([b]*5,values,color=color,s=12,alpha=.4)
            ax.plot([1,2,4,8],y,'o-',label=mode,color=color);ax.set_ylabel(label)
            ax.set_xlabel('Batch (I64 / O512)');ax.set_xticks([1,2,4,8]);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'batch_decode.png',dpi=180);plt.close(fig)
    traces=profiles['profiles']
    categories=sorted({name for v in traces.values() for cats in v['decode_trace']['kernel_categories_us'].values() for name in cats})
    fig,ax=plt.subplots(figsize=(11,5))
    labels=list(traces);bottom=np.zeros(len(labels))
    for name in categories:
        values=[]
        for key in labels:
            d=traces[key]['decode_trace']
            values.append(sum(cats.get(name,0) for cats in d['kernel_categories_us'].values())/d['actual_commits'])
        ax.bar(labels,values,bottom=bottom,label=name);bottom+=np.array(values)
    ax.set_ylabel('GPU kernel us / actual committed output token')
    ax.set_title('Separate steady decode traces; physical kernel totals, profiler enabled')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);fig.tight_layout();fig.savefig(out/'trace_decode.png',dpi=180);plt.close(fig)


def linear_overlap(left,right):
    from mtp_profile_analysis import union
    left,right=union(left),union(right)
    i=j=0
    total=0
    while i<len(left) and j<len(right):
        a,b=left[i]
        c,d=right[j]
        total+=max(0,min(b,d)-max(a,c))
        if b<=d:
            i+=1
        else:
            j+=1
    return total


def trace_detail_report(profiles,root,out):
    from bench_one_batch import kernel_category
    details={}
    api_rows=[]
    unknown_rows=[]
    sync_rows=[]
    for key,value in profiles['profiles'].items():
        record=value['decode_trace']
        path=Path(record['trace'])
        if not path.is_absolute():path=root/path
        events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
        physical=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
        intervals=[(e['ts'],e['ts']+e['dur']) for e in physical]
        runtime=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('cuda_runtime','cuda_driver')]
        cpu=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cpu_op']
        api=defaultdict(list)
        unknown=Counter()
        for e in runtime:api[e['name']].append(e)
        for e in physical:
            if e['cat']=='kernel' and kernel_category(e['name'])=='other':unknown[e['name']]+=e['dur']
        api_summary=[]
        for name,rows in sorted(api.items(),key=lambda item:-sum(e['dur'] for e in item[1])):
            total=sum(e['dur'] for e in rows)
            common=linear_overlap(intervals,[(e['ts'],e['ts']+e['dur']) for e in rows])
            api_summary.append(dict(name=name,calls=len(rows),cpu_us=total,gpu_overlap_us=common))
        longest=[]
        sync=[e for e in runtime if 'Synchronize' in e['name']]
        for e in sorted(sync,key=lambda e:-e['dur'])[:6]:
            parents=[p for p in cpu if p.get('pid')==e.get('pid') and p.get('tid')==e.get('tid')
                     and p['ts']<=e['ts'] and e['ts']+e['dur']<=p['ts']+p['dur']]
            chain=[p['name'] for p in sorted(parents,key=lambda p:-p['dur'])]
            longest.append(dict(api=e['name'],cpu_us=e['dur'],cpu_parent_chain=chain))
        details[key]=dict(runtime_api=api_summary,unknown_kernel_us=dict(unknown.most_common()),longest_sync_calls=longest)
        for row in api_summary[:5]:
            api_rows.append([key,row['name'],row['calls'],f"{row['cpu_us']/1000:.2f}",f"{row['gpu_overlap_us']/1000:.2f}"])
        for name,us in unknown.most_common(4):
            unknown_rows.append([key,name[:140].replace('|','/'),f'{us/1000:.2f}'])
        for row in longest[:3]:
            sync_rows.append([key,row['api'],f"{row['cpu_us']/1000:.2f}",' → '.join(row['cpu_parent_chain']) or '无aten父范围'])
    (out/'trace_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    sections=['# 稳态decode的CPU API及未分类kernel补充',
        'CPU API耗时包含等待GPU的区间，与GPU重合部分不能当作可消除host开销。这里只按trace中实际事件统计；没有启用Python栈，aten父范围不能单独证明某个具体Python调用点。',
        table(['配置','CUDA API','次数','CPU累计ms','与设备忙碌重合ms'],api_rows),
        '## 最长同步调用的aten父范围',table(['配置','API','CPU ms','外到内的aten范围'],sync_rows),
        '## other类别的最大kernel',
        '名称截取前140字符，完整名称及全部累计值保存在trace_details.json。这里主要用于检查copy/cast、reduction及逐元素计算是否遗漏，不把未知项自动归入GDN。',
        table(['配置','kernel名称前缀','累计ms'],unknown_rows)]
    (out/'trace_details.md').write_text('\n\n'.join(sections)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args();out=args.directory
    timing=json.loads((out/'results.json').read_text())
    profiles=json.loads((out/'profiles.json').read_text())
    root=Path(__file__).resolve().parents[1]
    groups=audit(timing,profiles,root)
    render(timing,profiles,groups,out)
    plots(timing,profiles,groups,out)
    trace_detail_report(profiles,root,out)
    print('Audited 90 samples, 6 diagnostic configurations, 12 trace artifacts; tables/CSV/plots written.')


if __name__=='__main__':
    main()
