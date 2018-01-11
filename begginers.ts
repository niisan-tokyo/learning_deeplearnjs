import {Array1D, Array2D, ENV, Graph, Scalar, Session, Tensor, NDArray, InCPUMemoryShuffledInputProviderBuilder, SGDOptimizer, CostReduction} from 'deeplearn'

async function mlBeginners() {
  const math = ENV.math

  {
    const matrixShape: [number, number] = [2, 3]
    const matrix = Array2D.new(matrixShape, [10, 20, 30, 40, 50, 60])
    const vector = Array1D.new([0, 1, 2])
    const result = math.matrixTimesVector(matrix, vector)

    console.log(matrix)
    console.log(vector)
    console.log('result shape: ', result.shape)
    console.log('result: ', await result.data())
    console.log(await matrix.data())
    console.log(await vector.data())
    console.log(matrix.shape)
  }

  {
    const graph = new Graph()

    // 入力のプレースホルダー作成
    const x: Tensor = graph.placeholder('x', [])

    // 各係数を定義
    const a: Tensor = graph.variable('a', Scalar.new(Math.random()))
    const b: Tensor = graph.variable('b', Scalar.new(Math.random()))
    const c: Tensor = graph.variable('c', Scalar.new(Math.random()))

    // 各項を定義
    const order2: Tensor = graph.multiply(a, graph.square(x))
    const order1: Tensor = graph.multiply(b, x)

    // 関数を定義
    // y = ax^2 + bx + c
    const y: Tensor = graph.add(graph.add(order2, order1), c)

    // y実測値のプレースホルダー作成
    const yLabel: Tensor = graph.placeholder('y label', [])

    // 誤差関数を定義
    const cost: Tensor = graph.meanSquaredCost(y, yLabel)

    // セッション生成
    const session = new Session(graph, math)

    await math.scope(async () => {
      // 現時点での計算グラフに4を入れた場合の値を出してみる
      let result: NDArray = session.eval(y, [{tensor: x, data: Scalar.new(4)}])
      console.log(await result.data())

      // 入力サンプルを定義
      const xs: Scalar[] = [
        Scalar.new(0), Scalar.new(1), Scalar.new(2), Scalar.new(3)
      ]

      // 出力のサンプルを定義
      const ys: Scalar[] = [
        Scalar.new(1.1), Scalar.new(5.9), Scalar.new(16.8), Scalar.new(33.9)
      ]

      // 入出力の組をシャッフルする機構を作成
      const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([xs, ys])
      const [xProvider, yProvider] = shuffledInputProviderBuilder.getInputProviders()

      // バッチの試行回数とバッチサイズを定義
      // バッチサイズは入力サンプルのサイズと同じ ( 普通のバッチ学習 )
      const NUM_BATCHES = 20
      const BATCH_SIZE = xs.length

      // 学習率を定義
      const LEARNING_RATE = 0.01

      // 最適化関数を定義。今回は普通の最速降下法
      const optimizer = new SGDOptimizer(LEARNING_RATE)


      console.log('Initial data: ')
      console.log(await session.eval(y, [{tensor: x, data: Scalar.new(4)}]).data())

      // 学習を実施
      for (let i = 0; i < NUM_BATCHES; i++) {
        const costValue = session.train(cost, [{tensor: x, data: xProvider}, {tensor: yLabel, data: yProvider}], BATCH_SIZE, optimizer, CostReduction.MEAN)
        console.log(`average const: ${costValue.get()}`)
      }

      result = session.eval(y, [{tensor: x, data: Scalar.new(4)}])
      console.log('result should be ~57.0: ')
      console.log(await result.data())
      const row_data = await result.data()
      console.log(row_data[0])
    })
  }
}

mlBeginners()
