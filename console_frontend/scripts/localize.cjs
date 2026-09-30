const fs = require('node:fs')
const path = require('node:path')
const ts = require('typescript')
const root = path.resolve(__dirname, '..')
const file = path.join(root, 'src', 'App.tsx')
const sourceText = fs.readFileSync(file, 'utf8')
const dictionary = fs.readFileSync(path.join(root, 'src', 'i18n.ts'), 'utf8')
const keys = new Set([...dictionary.matchAll(/^\s*'((?:\\'|[^'])+)':/gm)].map(match => match[1]))
const start = sourceText.indexOf('function PerformancePanel')
if (start < 0) throw new Error('Component start marker missing')
const source = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const call = value => ts.factory.createCallExpression(ts.factory.createIdentifier('t'), undefined, [ts.factory.createStringLiteral(value)])
let transformed = 0
const transformer = context => {
  const visit = node => {
    if (node.pos < start) return ts.visitEachChild(node, visit, context)
    if (ts.isJsxAttribute(node) && node.initializer && ts.isStringLiteral(node.initializer) && keys.has(node.initializer.text)) {
      transformed++
      return ts.factory.updateJsxAttribute(node, node.name, ts.factory.createJsxExpression(undefined, call(node.initializer.text)))
    }
    if (ts.isJsxText(node)) {
      const value = node.getText(source).trim()
      if (keys.has(value)) { transformed++; return ts.factory.createJsxExpression(undefined, call(value)) }
    }
    if (ts.isStringLiteral(node) && keys.has(node.text) && !(node.parent && ts.isCallExpression(node.parent) && node.parent.expression.getText(source) === 't')) { transformed++; return call(node.text) }
    return ts.visitEachChild(node, visit, context)
  }
  return fileNode => ts.visitNode(fileNode, visit)
}
const result = ts.transform(source, [transformer])
const output = ts.createPrinter({newLine: ts.NewLineKind.LineFeed}).printFile(result.transformed[0])
result.dispose()
fs.writeFileSync(file, output, 'utf8')
console.log(`Localized ${transformed} Chinese UI strings in App.tsx`)
