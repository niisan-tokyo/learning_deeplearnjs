import {Array1D, Array2D, ENV, Graph, Scalar, Session, Tensor, NDArray, InCPUMemoryShuffledInputProviderBuilder, SGDOptimizer, CostReduction, AdamOptimizer} from 'deeplearn'

const graph = new Graph()

// 学習用のデータセットを作成
const inputs: Array1D[] = []
const labels: Array1D[] = []
const indexes = Array(200).fill(0).map((v,i) => i - 99)
for (let i of indexes) {
  for (let j of indexes) {
    const x = j / 40
    const y = i / 40
    const input = Array1D.new([x, y])
    let yes = 0
    let no  = 1
    if (x * x + y * y > 1 && x * x + y * y < 4) {
      yes = 1
      no  = 0
    }
    const output = Array1D.new([yes, no])
    inputs.push(input)
    labels.push(output)
  }
}

const math = ENV.math

const input_dim = 2
const label_dim = 2

// 計算用の多層パーセプトロンを生成
const x0: Tensor = graph.placeholder('input', [input_dim])

const a1: Tensor = graph.variable('a1', Array2D.randNormal([64, input_dim]))
const b1: Tensor = graph.variable('b1', Array1D.zeros([64]))
const y1: Tensor = graph.matmul(a1, x0)
const x1: Tensor = graph.relu(graph.add(y1, b1))

const a2: Tensor = graph.variable('a2', Array2D.randNormal([128, 64]))
const b2: Tensor = graph.variable('b2', Array1D.zeros([128]))
const y2: Tensor = graph.matmul(a2, x1)
const x2: Tensor = graph.tanh(graph.add(y2, b2))

const a3: Tensor = graph.variable('a3', Array2D.randNormal([label_dim, 128]))
const b3: Tensor = graph.variable('b3', Array1D.zeros([label_dim]))
const y3: Tensor = graph.matmul(a3, x2)

const x4: Tensor = graph.softmax(graph.add(y3, b3))
const y : Tensor = graph.placeholder('label', [label_dim])

const cost: Tensor = graph.softmaxCrossEntropyCost(x4, y)

const session = new Session(graph, math)

async function train() {

  // ここから学習の設定及び実行
  await math.scope(async () => {
    const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([inputs, labels])
    const [xProvider, yProvider] = shuffledInputProviderBuilder.getInputProviders()

    const num = <HTMLInputElement>document.getElementById('num_batch')
    console.log(num)

    const NUM_BATCHES = +num.value // number へのキャスト
    const BATCH_SIZE = 128

    // Learning Parameter
    const LEARNING_RATE = 0.01
    const BETA_1        = 0.9
    const BETA_2        = 0.999

    //const optimizer = new SGDOptimizer(LEARNING_RATE)
    const optimizer = new AdamOptimizer(LEARNING_RATE, BETA_1, BETA_2)

    // 学習開始
    // BATCH_SIZE != データ数なので、NUM_BATCHES がいわゆるepoch数とは違うことに注意
    for (let i = 0; i < NUM_BATCHES; i++) {
      const costValue = session.train(
        cost,
        [{tensor: x0, data: xProvider}, {tensor: y, data: yProvider}],
        BATCH_SIZE,
        optimizer,
        CostReduction.MEAN
      )
      if (i % 100 == 1) {
        console.log(`average cost: ${costValue.get()} in ${i}`)
      }
    }

    console.log('finish learning!!')
  })
}

async function draw() {

  // 学習が完了したら、canvasで領域を図示する
  const canvas = document.getElementById('plot_area')
  console.log(canvas)
  drawLine(canvas, [
    {xi: 100, yi: 0, xf: 100, yf: 400},
    {xi: 200, yi: 0, xf: 200, yf: 400},
    {xi: 300, yi: 0, xf: 300, yf: 400},
    {xi: 0, yi: 100, xf: 400, yf: 100},
    {xi: 0, yi: 200, xf: 400, yf: 200},
    {xi: 0, yi: 300, xf: 400, yf: 300},
  ])

  // 学習済みのモデルに適当な座標を読み込ませて、領域内だと判断したら、プロットする
  for (let i = 0; i < 40; i++) {
    const circles = []
    for (let j = 0; j < 40; j++) {
      const xe = (j - 20) / 10
      const ye = (i - 20) / 10
      const input_data = Array1D.new([xe, ye])
      const res = session.eval(x4, [{tensor: x0, data: input_data}])
      const row_data = await res.data()
      if (row_data[0] > 0.5) {
        plot_canvas(canvas, j, i)
      }
    }

  }
}

async function reset() {
  const canvas = <HTMLCanvasElement>document.getElementById('plot_area')
  let ctx = canvas.getContext('2d')
  ctx.beginPath()
  ctx.clearRect(0, 0, 400, 400)
}

function drawLine(canvas, rules) {
  for (let rule of rules) {
    let ctx = canvas.getContext('2d')
    ctx.beginPath()
    ctx.moveTo(rule.xi, rule.yi)
    ctx.lineTo(rule.xf, rule.yf)
    ctx.closePath()
    ctx.stroke()
  }
}

function plot_canvas(canvas, x, y) {
  const ctx = canvas.getContext('2d')
  ctx.beginPath()
  ctx.fillRect(x * 10, y * 10, 10, 10)
}

const train_button = document.getElementById('train')
train_button.addEventListener('click', train)

const draw_button = document.getElementById('draw')
draw_button.addEventListener('click', draw)

const reset_button = document.getElementById('reset')
reset_button.addEventListener('click', reset)
