import type { TensorValue } from './types'
import { addTensorsExact, elementwiseTensors, multiplyTensors, reduceToShape, scalarValue, scaleTensor, subtractTensors, zeroLike } from './tensor'

type Expr =
  | { kind: 'number'; value: number }
  | { kind: 'input'; index: number }
  | { kind: 'negate'; child: Expr }
  | { kind: 'binary'; op: '+' | '-' | '*' | '/' | '^'; left: Expr; right: Expr }

export interface ParsedArithmetic { expression: Expr; inputCount: number }

/** Small, explicit expression language: numbers, x1…x64, parentheses and arithmetic. */
export function parseArithmetic(source: string): ParsedArithmetic {
  const tokens = source.match(/(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|x\d+|[^\s]/g) ?? []
  let cursor = 0
  let highestInput = 0
  const peek = () => tokens[cursor]
  const take = () => tokens[cursor++]
  const fail = (message: string): never => { throw new Error(message) }
  const primary = (): Expr => {
    const token = take()
    if (token === '(') {
      const inner = additive()
      if (take() !== ')') fail('Close the parenthesis.')
      return inner
    }
    if (/^x[1-9]\d*$/.test(token ?? '')) {
      const index = Number(token.slice(1)) - 1
      if (index >= 64) fail('Use inputs x1 through x64.')
      highestInput = Math.max(highestInput, index + 1)
      return { kind: 'input', index }
    }
    if (/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(token ?? '')) {
      const value = Number(token)
      if (!Number.isFinite(value)) fail('Use finite numbers.')
      return { kind: 'number', value }
    }
    return fail(token ? `Unexpected “${token}”.` : 'Enter an expression, such as x1 * x2 + 1.')
  }
  const power = (): Expr => {
    const left = primary()
    return peek() === '^' ? (take(), { kind: 'binary', op: '^', left, right: unary() }) : left
  }
  const unary = (): Expr => {
    if (peek() === '+') { take(); return unary() }
    if (peek() === '-') { take(); return { kind: 'negate', child: unary() } }
    return power()
  }
  const multiplicative = (): Expr => {
    let left = unary()
    while (peek() === '*' || peek() === '/') {
      const op = take() as '*' | '/'
      left = { kind: 'binary', op, left, right: unary() }
    }
    return left
  }
  const additive = (): Expr => {
    let left = multiplicative()
    while (peek() === '+' || peek() === '-') {
      const op = take() as '+' | '-'
      left = { kind: 'binary', op, left, right: multiplicative() }
    }
    return left
  }
  const expression = additive()
  if (cursor < tokens.length) fail(`Unexpected “${peek()}”.`)
  if (highestInput === 0) fail('Reference at least one input, such as x1.')
  const referenced = new Set<number>()
  const visit = (expr: Expr): void => {
    if (expr.kind === 'input') referenced.add(expr.index)
    if (expr.kind === 'negate') visit(expr.child)
    if (expr.kind === 'binary') { visit(expr.left); visit(expr.right) }
  }
  visit(expression)
  for (let index = 0; index < highestInput; index++) {
    if (!referenced.has(index)) fail(`Use x${index + 1} before x${highestInput}; input numbers must be consecutive.`)
  }
  const numericConstant = (expr: Expr): boolean => expr.kind === 'number' || (expr.kind === 'negate' && numericConstant(expr.child))
  const checkPower = (expr: Expr): void => {
    if (expr.kind === 'binary') {
      if (expr.op === '^' && !numericConstant(expr.right)) fail('The exponent must be a number, such as x1^2.')
      checkPower(expr.left); checkPower(expr.right)
    }
    if (expr.kind === 'negate') checkPower(expr.child)
  }
  checkPower(expression)
  return { expression, inputCount: highestInput }
}

export function arithmeticInputCount(source: string): number { return parseArithmetic(source).inputCount }

/** Add a real input to an editable expression, preserving its top-level operation. */
export function appendArithmeticInput(source: string): string {
  const parsed = parseArithmetic(source)
  if (parsed.inputCount >= 64) throw new Error('Arithmetic expressions support up to 64 inputs.')
  const operator = parsed.expression.kind === 'binary' && parsed.expression.op === '*' ? '*' : '+'
  return `${source.trim()} ${operator} x${parsed.inputCount + 1}`
}

interface Evaluated { value: TensorValue; backward: (gradient: TensorValue) => void }

export function evaluateArithmetic(source: string, inputs: TensorValue[], gradient?: TensorValue): { value: TensorValue; gradients: TensorValue[] } {
  const parsed = parseArithmetic(source)
  if (inputs.length !== parsed.inputCount) throw new Error(`Expression needs ${parsed.inputCount} inputs.`)
  const gradients = inputs.map(zeroLike)
  const evaluate = (expr: Expr): Evaluated => {
    if (expr.kind === 'number') return { value: scalarValue(expr.value), backward: () => undefined }
    if (expr.kind === 'input') return { value: inputs[expr.index], backward: g => { gradients[expr.index] = addTensorsExact(gradients[expr.index], reduceToShape(g, inputs[expr.index].shape)) } }
    if (expr.kind === 'negate') {
      const child = evaluate(expr.child)
      return { value: scaleTensor(child.value, -1), backward: g => child.backward(scaleTensor(g, -1)) }
    }
    const left = evaluate(expr.left), right = evaluate(expr.right)
    const binary = (fn: (a: number, b: number) => number) => elementwiseTensors([left.value, right.value], ([a, b]) => fn(a, b))
    const send = (operand: Evaluated, g: TensorValue) => operand.backward(reduceToShape(g, operand.value.shape))
    if (expr.op === '+') return { value: binary((a, b) => a + b), backward: g => { send(left, g); send(right, g) } }
    if (expr.op === '-') return { value: subtractTensors(left.value, right.value), backward: g => { send(left, g); send(right, scaleTensor(g, -1)) } }
    if (expr.op === '*') return { value: multiplyTensors(left.value, right.value), backward: g => { send(left, multiplyTensors(g, right.value)); send(right, multiplyTensors(g, left.value)) } }
    if (expr.op === '/') return {
      value: binary((a, b) => a / b),
      backward: g => {
        send(left, multiplyTensors(g, binary((_, b) => 1 / b)))
        send(right, multiplyTensors(g, binary((a, b) => -a / (b * b))))
      },
    }
    const constantValue = (part: Expr): number => part.kind === 'number' ? part.value : part.kind === 'negate' ? -constantValue(part.child) : NaN
    const exponent = constantValue(expr.right)
    return {
      value: elementwiseTensors([left.value], ([a]) => a ** exponent),
      backward: g => send(left, multiplyTensors(g, elementwiseTensors([left.value], ([a]) => exponent === 0 ? 0 : exponent * a ** (exponent - 1)))),
    }
  }
  const result = evaluate(parsed.expression)
  if (!result.value.data.every(Number.isFinite)) throw new Error('Arithmetic result is not finite; check division and powers.')
  if (gradient) result.backward(gradient)
  return { value: result.value, gradients }
}
