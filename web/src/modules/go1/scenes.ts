export type Go1SceneKind='turn'|'uturn'|'slope'|'duck'|'jump'|'flat';
export interface Go1Scene {format:'go1-scene/v1';id:string;title:string;kind:Go1SceneKind;createdAt:string;criteria?:Record<string,number>;parameters:Record<string,number>;}
export const sceneKinds={flat:'空旷场地',turn:'拐弯',uturn:'窄路掉头',slope:'越过小坡',duck:'俯身穿过横梁',jump:'跳过低横梁'};
export const sceneFields:Record<Go1SceneKind,{key:string;label:string;min:number;max?:number;value:number}[]>={
 flat:[{key:'forward_m_s',label:'前进指令（m/s）',min:-2,max:2,value:.3},{key:'yaw_rate_rad_s',label:'转向指令（rad/s）',min:-1,max:1,value:0}],
 turn:[{key:'angle_deg',label:'拐弯角度（°）',min:45,max:135,value:90},{key:'road_width_m',label:'道路宽度（m）',min:.6,value:.8},{key:'distance_m',label:'转弯前距离（m）',min:.8,value:1.2}],
 uturn:[{key:'road_width_m',label:'道路宽度（m）',min:.6,value:.8},{key:'turn_at_m',label:'掉头位置（m）',min:.8,value:1.2}],
 duck:[{key:'distance_m',label:'横梁距离（m）',min:.8,value:1.2},{key:'height_m',label:'梁下净高（m）',min:.26,max:.30,value:.28}],
 jump:[{key:'distance_m',label:'横梁距离（m）',min:.8,value:1.2},{key:'height_m',label:'横梁高度（m）',min:.16,max:.24,value:.20}],
 slope:[{key:'angle_deg',label:'坡度（°）',min:3,max:15,value:8},{key:'length_m',label:'坡长（m）',min:.8,value:1.2}],
};
export function sceneSummary(scene:Go1Scene){return sceneFields[scene.kind].map(field=>`${field.label} ${scene.parameters[field.key]}`).join(' · ');}
export function sceneSuccess(scene:Go1Scene){if(scene.kind==='flat')return '记录运动轨迹';const c=sceneCriteria(scene);return (scene.kind==='turn'?`转向 ${scene.parameters.angle_deg}° 后前进至少 ${c.exit_distance_m} m，方向误差 ≤ ${c.heading_tolerance_deg}°`:scene.kind==='uturn'?`完成 180° 掉头并返回起点 ${c.return_tolerance_m} m 范围，方向误差 ≤ ${c.heading_tolerance_deg}°`:scene.kind==='duck'?`俯身从梁下通过并额外前进 ${c.clearance_m} m`:scene.kind==='jump'?`四脚离地越梁，落地后额外前进 ${c.clearance_m} m`:`机器人整体通过坡顶并额外前进 ${c.clearance_m} m`)+`，稳定 ${c.stable_seconds} 秒；不跌倒、不碰撞、不越界`;}
export function validateSceneInput(title:string,kind:Go1SceneKind,parameters:Record<string,number>,criteria?:Record<string,number>){
 if(!title.trim()||[...title].length>80)throw Error('请填写80字以内的场景名称');
 const fields=sceneFields[kind];if(!fields||Object.keys(parameters).length!==fields.length||fields.some(field=>!Number.isFinite(parameters[field.key])||parameters[field.key]<field.min||(field.max!==undefined && parameters[field.key]>field.max)))throw Error('场景参数超出范围');
 return {title:title.trim(),kind,parameters,criteria:kind==='flat'?undefined:validateCriteria(kind,parameters,criteria)};
}

export const criterionFields:Record<Go1SceneKind,{key:string;label:string;min:number;max?:number;value:number}[]>={
 flat:[],
 turn:[{key:'exit_distance_m',label:'转弯后前进距离（m）',min:.1,value:.8},{key:'heading_tolerance_deg',label:'方向允许误差（°）',min:1,max:45,value:15},{key:'stable_seconds',label:'保持稳定时间（秒）',min:.02,max:10,value:1}],
 uturn:[{key:'return_tolerance_m',label:'返回起点范围（m）',min:.01,value:.15},{key:'heading_tolerance_deg',label:'方向允许误差（°）',min:1,max:45,value:15},{key:'stable_seconds',label:'保持稳定时间（秒）',min:.02,max:10,value:1}],
 duck:[{key:'clearance_m',label:'过梁额外距离（m）',min:0,value:0},{key:'stable_seconds',label:'保持稳定时间（秒）',min:.02,max:10,value:1}],
 jump:[{key:'clearance_m',label:'过梁额外距离（m）',min:0,value:0},{key:'stable_seconds',label:'保持稳定时间（秒）',min:.02,max:10,value:1}],
 slope:[{key:'clearance_m',label:'坡顶额外通过距离（m）',min:0,value:0},{key:'stable_seconds',label:'保持稳定时间（秒）',min:.02,max:10,value:1}],
};
export function sceneCriteria(scene:Pick<Go1Scene,'kind'|'criteria'>):Record<string,number>{return {...Object.fromEntries(criterionFields[scene.kind].map(field=>[field.key,field.value])),...scene.criteria};}
export function validateCriteria(kind:Go1SceneKind,parameters:Record<string,number>,criteria?:Record<string,number>){
 const fields=criterionFields[kind],value=criteria ?? sceneCriteria({kind});
 if(Object.keys(value).length!==fields.length || fields.some(field=>!Number.isFinite(value[field.key])||value[field.key]<field.min||(field.max!==undefined&&value[field.key]>field.max)))throw Error('通过条件超出范围');
 if(kind==='uturn'&&value.return_tolerance_m>parameters.turn_at_m/2)throw Error('返回起点范围不能超过掉头距离的一半');
 return value;
}
