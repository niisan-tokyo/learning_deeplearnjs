import {Array1D, Array2D, ENV, Graph, Scalar, Session, Tensor, NDArray, InCPUMemoryShuffledInputProviderBuilder, SGDOptimizer, CostReduction} from 'deeplearn'

const a = Scalar.new(4)
const b = Array1D.new([2, 3])
const c = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]])

async function calc() {
  const math = ENV.math

  console.log(await a.data())// Float32Array [4]
  console.log(await b.data())// Float32Array(2) [2, 3]
  console.log(await c.data())// Float32Array(6) [1, 2, 3, 4, 5, 6]

  const d = math.add(a, b)
  const e = math.add(a, c)
  console.log(await d.data())// Float32Array(2) [6, 7]
  console.log(await e.data())// Float32Array(6) [5, 6, 7, 8, 9, 10]

  const f = math.add(c, b)
  console.log(await f.data())// Float32Array(6) [3, 5, 5, 7, 7, 9]

  const g = math.multiply(b, a)
  const h = math.multiply(c, a)
  const i = math.multiply(c, b)

  console.log(await g.data())// Float32Array(2) [8, 12]
  console.log(await h.data())// Float32Array(6) [4, 8, 12, 16, 20, 24]
  console.log(await i.data())// Float32Array(6) [2, 6, 6, 12, 10, 18]

  const j = math.matrixTimesVector(c, b)
  console.log(await j.data())// 
}

calc()
