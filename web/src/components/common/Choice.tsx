import { Select, SelectTrigger, SelectValue, SelectIcon, SelectPortal, SelectPositioner, SelectContent, SelectList, SelectItem, SelectItemText } from '@odyssey/ui/primitives/select';

export function Choice({id,label,value,options,onChange,disabled=false,placeholder='没有可选模型'}: {
  id:string; label:string; value:string; options:{value:string;label:string}[]; onChange:(value:string)=>void; disabled?:boolean; placeholder?:string;
}) {
  return <Select value={value} onValueChange={value=>{if(value!==null) onChange(String(value));}} disabled={disabled}>
    <SelectTrigger id={id} aria-label={label}><SelectValue>{options.find(option=>option.value===value)?.label ?? placeholder}</SelectValue><SelectIcon /></SelectTrigger>
    <SelectPortal><SelectPositioner><SelectContent width="trigger"><SelectList>
      {options.map(option=><SelectItem key={option.value} value={option.value}><SelectItemText>{option.label}</SelectItemText></SelectItem>)}
    </SelectList></SelectContent></SelectPositioner></SelectPortal>
  </Select>;
}
