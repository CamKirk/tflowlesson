var tf = require('@tensorflow/tfjs');

var a = tf.variable(tf.scalar(1));
var b = tf.variable(tf.scalar(1));
var c = tf.variable(tf.scalar(1));

function predict(x) {
    return a.mul(x.square())
      .add(b.mul(x))
      .add(c)
}

function loss(prediction, labels) {
  const err = prediction.sub(labels).square().mean();
  //console.log("new error is: ");
  // err.print();
  return err
}


var numIterations = 200;
var learningRate= 0.01;
var optimizer = tf.train.sgd(learningRate);

function train(xs,ys, numIterations) {
  for (var i = 0; i < numIterations; i++) {
    optimizer.minimize(function () {
      var pred = predict(xs);
      return loss(pred,ys);
    })
  }
}

var xs = [1,2,3,4];
var ys = [7,3,1,6];

xs = tf.tensor1d(xs);
ys = tf.tensor1d(ys);

function learning() {


  console.log('pre-train prediction');
  predict(xs).print()

  console.log('train');
  train(xs, ys, numIterations);

  console.log('post-train predictions')
  predict(xs).print()

  console.log('prediction for unlabeled feature')
  predict(tf.scalar(5)).print()
}

learning();
