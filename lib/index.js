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
let a = tf.tidy(() => { 
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

let filter = tf.tensor([
  [0, 1, 0],
  [1,-4, 1],
  [0, 1, 0]
]).reshape([3,3,1,1])
filter = filter.mul(0.1);
const surface = { name: 'Heatmap', tab: 'Charts' };


const nt = 10;
let out;
let vis_mat;
let out_padded;

for (let it=0; it<nt; it++ ) {
  out = tf.conv2d(a,filter,1,'valid');
  out_padded = tf.pad(out,[[1,1], [1,1], [0,0]]);
  a = a.add(out_padded);
  vis_mat = a.squeeze();
  tfvis.render.heatmap(surface, {values: vis_mat});
}


// Clean memory
a.dispose();
filter.dispose();
out.dispose();
out_padded.dispose();
vis_mat.dispose();