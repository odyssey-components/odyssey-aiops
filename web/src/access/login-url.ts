const base=(import.meta.env.BASE_URL ?? '/').replace(/\/+$/,'');
const login=base+'/login';

export function loginURL(location:Location) {
 return login+'?'+new URLSearchParams({returnTo:location.pathname+location.search+location.hash});
}

export function loginDestination(location:Location) {
 const value=new URLSearchParams(location.search).get('returnTo');
 if(value?.startsWith('/') && !value.startsWith('//') && URL.canParse(value,location.origin)) {
  const target=new URL(value,location.origin);
  if(target.origin===location.origin && !target.pathname.startsWith("//") &&
     (target.pathname===base || target.pathname.startsWith(base+'/')) &&
     target.pathname.replace(/\/+$/,'')!==login) return target.pathname+target.search+target.hash;
 }
 return base+'/';
}
