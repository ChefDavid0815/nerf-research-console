const fs=require('node:fs'),path=require('node:path'),ts=require('typescript')
const file=path.resolve(__dirname,'../src/App.tsx'),raw=fs.readFileSync(file,'utf8')
const source=ts.createSourceFile(file,raw,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX)
const start=raw.indexOf('function PerformancePanel')
const missing=new Set()
function visit(node){
  if(node.pos>=start){
    if(ts.isStringLiteral(node)&&/\p{Script=Han}/u.test(node.text)&&!(node.parent&&ts.isCallExpression(node.parent)&&node.parent.expression.getText(source)==='t'))missing.add(`string: ${node.text}`)
    if(ts.isJsxText(node)&&/\p{Script=Han}/u.test(node.getText(source)))missing.add(`jsx: ${node.getText(source).trim()}`)
    if((ts.isTemplateHead(node)||ts.isTemplateMiddle(node)||ts.isTemplateTail(node)||ts.isNoSubstitutionTemplateLiteral(node))&&/\p{Script=Han}/u.test(node.text))missing.add(`template: ${node.text}`)
  }
  ts.forEachChild(node,visit)
}
visit(source)
console.log([...missing].join('\n'))
