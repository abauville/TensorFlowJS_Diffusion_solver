import * as tf from '@tensorflow/tfjs';
import { loadGraphModel, StringToHashBucketFast } from '@tensorflow/tfjs';
import * as p5 from 'p5';

function getIniTensor(nx=20, ny=20, sigma=0.2) {
  const x = tf.linspace(-1.0, 1.0, nx);
  const y = tf.linspace(-1.0, 1.0, ny);
  let X;
  let Y;
  [X, Y] = tf.meshgrid(x, y);
  let a = tf.tidy(() => { 
    // 2D gaussian
    return ( 
      X.square().mul(-0.5/(sigma*sigma)).exp().mul(
        Y.square().mul(-0.5/(sigma*sigma)).exp()
      )
    ).reshape([nx,ny,1]);
  });
  // a.print();
  x.dispose();
  y.dispose();
  X.dispose();
  Y.dispose();
  return a
}




function runStep(A, filter) {
  const out = tf.conv2d(A,filter,1,'valid');
  const out_padded = tf.pad(out,[[1,1], [1,1], [0,0]]);
  A = A.add(out_padded)
  out.dispose();
  out_padded.dispose();
  return A;
}





let T = getIniTensor(500, 500, 0.1);
let img;
const HEIGHT = 500
const WIDTH = HEIGHT;
let counter = 0
const sketch = (p) => {
  
  p.setup = () => {
    p.createCanvas(WIDTH, HEIGHT);
    p.noStroke();    
    img = p.createImage(T.shape[0], T.shape[1]);

  }
  p.draw = () => {
    console.log("c: ", counter);
    counter += 1;
    for (let i=0; i<50; i++) {
      T = runStep(T, filter);
    }
    // console.log("Diffuse end")

    // console.log("Plot start")
    plotImage(p);
    // console.log("Plot end")
    
    // console.log(img);
  }
}


function plotSquare(p) {
  const square_size = WIDTH/T.shape[0]
    T.array().then((array) => {
      for (let i = 0; i < img.width; i++) {
        for (let j = 0; j < img.height; j++) {
          p.fill(array[i][j][0]*255);
          p.square(square_size*i, square_size*j, square_size)
        }
      }
    });
}

function plotImage(p) {
  img.loadPixels();
  let val;
  T.array().then((array) => {
    for (let i = 0; i < img.width; i++) {
      for (let j = 0; j < img.height; j++) {
        val = array[i][j][0]*255
        img.set(i, j, p.color(val, val, val));
      }
    }
    img.updatePixels();
    p.image(img, 17, 17);
  });
}



let filter = tf.tensor([
  [0, 1, 0],
  [1,-4, 1],
  [0, 1, 0]
]).reshape([3,3,1,1])
filter = filter.mul(0.249);
const sketchP = new p5(sketch);