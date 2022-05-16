import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// const a = tf.tensor([[1, 2], [3, 4]]).reshape([2,2,1]);
const nx = 15;
const ny = 15;
const x = tf.linspace(-1.0, 1.0, nx);
const y = tf.linspace(-1.0, 1.0, ny);
let X;
let Y;
[X, Y] = tf.meshgrid(x, y);
const sigma = 0.2;
const a = tf.tidy(() => { 
  // 2D gaussian
  return ( 
    X.square().mul(-0.5/(sigma*sigma)).exp().mul(
      Y.square().mul(-0.5/(sigma*sigma)).exp()
    )
  ).reshape([nx,ny,1]);
});
console.log(a);
// a.print();
x.dispose();
y.dispose();
X.dispose();
Y.dispose();

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


// // Render to visor
const surface = { name: 'Heatmap', tab: 'Charts' };

// // tfvis.render.heatmap(surface, {values: a.reshape([5,5])});
const vis_mat = out.squeeze()
console.log(vis_mat.shape);
tfvis.render.heatmap(surface, {values: vis_mat});

// Clean memory
a.dispose();
filter.dispose();
out.dispose();
vis_mat.dispose();