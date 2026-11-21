import type {TrainingMetric} from './training-contract';

export function chartMetrics(metrics:readonly TrainingMetric[]):TrainingMetric[] {
  const unique=new Map<number,TrainingMetric>();
  for(const point of metrics) {
    if(Number.isInteger(point.epoch) && point.epoch>=0 && Number.isInteger(point.total) && point.epoch<=point.total
      && Number.isFinite(point.successRate) && point.successRate>=0 && point.successRate<=1
      && Number.isInteger(point.evaluatedMaps) && point.evaluatedMaps>0) unique.set(point.epoch,point);
  }
  return [...unique.values()].sort((left,right)=>left.epoch-right.epoch);
}
export function metricSummary(points:readonly TrainingMetric[]) {
  if(!points.length) return null;
  const best=points.reduce((peak,point)=>point.successRate>peak.successRate?point:peak);
  const latest=points[points.length-1],baseline=points.find(point=>point.epoch===0);
  return {latest,best,change:baseline?(latest.successRate-baseline.successRate)*100:null};
}
export function nearestMetric(points:readonly TrainingMetric[],epoch:number) {
  return points.reduce<TrainingMetric|undefined>((nearest,point)=>!nearest || Math.abs(point.epoch-epoch)<Math.abs(nearest.epoch-epoch)?point:nearest,undefined);
}
export function metricsCsv(points:readonly TrainingMetric[]) {
  return 'epoch,total,success_rate,evaluated_maps,split,training_loss,training_action_accuracy\n'+points.map(point=>[
    point.epoch,point.total,point.successRate,point.evaluatedMaps,point.split,point.loss??'',point.accuracy??'',
  ].join(',')).join('\n')+'\n';
}
