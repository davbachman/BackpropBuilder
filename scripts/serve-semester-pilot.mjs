/** Local-only pilot server. Accepts results only from this server's browser origin. */
import {createServer} from 'vite'
import {writeFile,mkdir} from 'node:fs/promises'
await mkdir('output/semester-pilot/browser',{recursive:true})
const server=await createServer({server:{host:'127.0.0.1',port:5174,strictPort:true},plugins:[{name:'pilot-results',configureServer(server){server.middlewares.use('/pilot-result',async(req,res,next)=>{
 if(req.method!=='POST')return next()
 if(req.headers.origin!=='http://127.0.0.1:5174'){res.statusCode=403;return res.end()}
 try{let body='';for await(const chunk of req){body+=chunk;if(body.length>80_000_000)throw Error('Too large')}
 const parsed=JSON.parse(body);if(!/^[a-zA-Z0-9.-]+$/.test(parsed.name))throw Error('Invalid result name')
 await writeFile('output/semester-pilot/browser/'+parsed.name+'.json',JSON.stringify(parsed,null,2));res.end('saved')
 }catch(e){res.statusCode=400;res.end(String(e))}
})}}]})
await server.listen();server.printUrls()
