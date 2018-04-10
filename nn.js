/* Params */
const NUM_CLASSES = 3;


/**
 * Train/Fit wrapper
 */
function trainModel(xTrain, yTrain, xTest, yTest, options) {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'}));

  const optimizer = tf.train.adam(options.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const lossValues = [];
  const accuracyValues = [];

  let promise = model.fit(xTrain, yTrain, {
    epochs: options.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (epoch % 10 === 0) {
          console.log('processed', epoch, logs);
        }
        options.onEpochEnd(epoch, logs);
        tf.nextFrame();
      },
      onTrainEnd: (logs, a) => {
        options.onTrainEnd();
      }
    }
  });

  promise.then( data => {
    /*
    console.log('done promise', data);
    const testInput = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    const testTensor = tf.tensor2d(testInput, [1, 13]);
    let x = data.model.predict(testTensor);
    console.log(x.dataSync());
    */
  });
  return model;
}


