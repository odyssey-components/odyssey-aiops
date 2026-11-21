export function go1Evaluation(value:unknown):string {
 if(!value||typeof value!=='object')return '未评估';
 const v=value as Record<string,unknown>;
 const labels:Record<string,string>={observed:'诊断已录制',success:'通过',fallen:'未通过 · 跌倒',fall:'未通过 · 跌倒',collision:'未通过 · 碰撞',bypass:'未通过 · 偏离道路',timeout:'未通过 · 超时'};
 if(v.format==='go1-command-evaluation/v2' && typeof v.steps==='number' && v.steps>0){
  return v.capability_verified===true?`能力已验证 · ${v.steps} 步`:`已记录诊断 · ${v.steps} 步，能力未验证`;
 }
 if(v.outcome==='observed' && typeof v.steps==='number')return `诊断已录制 · ${v.steps} 步`;
 if(typeof v.outcome==='string')return labels[v.outcome] ?? '未评估';
 if(typeof v.total==='number'&&v.total>0&&typeof v.successes==='number')return `${v.successes===v.total?'通过':'未通过'} · ${v.successes}/${v.total}`;
 if(typeof v.episode_ends==='number'&&v.episode_ends>0)return '未通过 · 跌倒';
 return '未评估';
}

export function go1SimulationEvaluation(state:string,result:unknown):string {
 if(result)return go1Evaluation(result);
 if(state==='failed')return '执行失败 · 无评估结果';
 if(state==='cancelled')return '已取消 · 无评估结果';
 return state==='succeeded'?'该任务未保存评估结果':'待评估';
}
