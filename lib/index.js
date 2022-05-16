import * as tf from '@tensorflow/tfjs';

// const a = tf.tensor([[1, 2], [3, 4]]).reshape([2,2,1]);
const nx = 5;
const ny = 5;
const a = tf.zeros([nx,ny,1])
const filter = tf.tensor([
  [0, 1, 0],
  [1,-2, 1],
  [0, 1, 0]
]).reshape([3,3,1,1])
console.log(a.shape)
// a.print()
console.log(filter.shape)
// filter.print()
const out = tf.conv2d(a,filter,1,'valid');
console.log("OUT =========")
console.log(out.shape);
out.print();

// Clean memory
a.dispose();
filter.dispose();
out.dispose();
