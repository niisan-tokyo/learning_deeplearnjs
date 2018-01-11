import {Array2D, Tensor, Graph, ENV} from 'deeplearn';

// async function runExample() {
//   const math = new NDArrayMathGPU();
//   const a = Array1D.new([1, 2, 3]);
//   const b = Scalar.new(2);
//
//   const result = math.add(a, b);
//
//   // Option 1: With async/await.
//   // Caveat: in non-Chrome browsers you need to put this in an async function.
//   console.log(await result.data());  // Float32Array([3, 4, 5])
//
//   // Option 2: With a Promise.
//   result.data().then(data => console.log(data));
//
//   // Option 3: Synchronous download of data.
//   // This is simpler, but blocks the UI until the GPU is done.
//   console.log(result.dataSync());
// }
//
// runExample();
//

const g = new Graph()
const math = ENV.math;

const x: Tensor = g.placeholder('x', [])
const W1data = Array2D.randNormal([16, 1])
const b1data = Array2D.zeros([16, 1])
// const result = math.add(W1data, b1data)
console.log(W1data)
console.log(b1data)
// console.log(result)

//async function sample() {
//  console.log(await result.data())
//}
//sample()


const W1: Tensor = g.variable('W1', W1data)
const b1: Tensor = g.variable('b1', b1data)
console.log(W1)
console.log(b1)
