import {useState} from 'react';
import type {TablePaginationOptions} from '@odyssey/ui/primitives/table';

export interface PageInfo {page:number;size:number;total:number}

export function tablePagination(info:PageInfo|undefined,onPageChange:(page:number)=>void,onPageSizeChange:(size:number)=>void,disabled=false):TablePaginationOptions|undefined {
 if(!info)return undefined;
 return {page:info.page,pageSize:info.size,total:info.total,onPageChange,onPageSizeChange,disabled};
}

export function useLocalTablePagination(total:number) {
 const [page,setPage]=useState(1),[size,setSize]=useState(20);
 const current=Math.min(page,Math.max(1,Math.ceil(total/size)));
 const start=(current-1)*size;
 return {
  options:{page:current,pageSize:size,total,onPageChange:setPage,onPageSizeChange:(next:number)=>{setSize(next);setPage(1);}},
  slice<T>(rows:readonly T[]){return rows.slice(start,start+size);},
 };
}
