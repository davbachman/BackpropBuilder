import type { TensorValue } from '../domain/types'

const zeros = (shape: number[]): TensorValue => ({ shape, data: Array(shape.reduce((a,b) => a*b, 1)).fill(0) })

function convolutionShape(input: TensorValue, weights: TensorValue, bias: TensorValue) {
  const [height, width, channels] = input.shape, [filters, kh, kw, wc] = weights.shape
  if (input.shape.length !== 3 || weights.shape.length !== 4 || wc !== channels || bias.shape.length !== 1 || bias.shape[0] !== filters || height < kh || width < kw) throw new Error('Convolution expects H×W×C input, F×KH×KW×C filters, and F biases.')
  return { height, width, channels, filters, kh, kw, oh: height-kh+1, ow: width-kw+1 }
}

/** Valid stride-one cross-correlation, the convention used by convolutional networks. */
export function conv2d(input: TensorValue, weights: TensorValue, bias: TensorValue): TensorValue {
  const { width, channels, filters, kh, kw, oh, ow } = convolutionShape(input, weights, bias)
  const output = zeros([oh, ow, filters])
  for (let y=0; y<oh; y++) for (let x=0; x<ow; x++) for (let f=0; f<filters; f++) {
    let sum = bias.data[f]
    for (let ky=0; ky<kh; ky++) for (let kx=0; kx<kw; kx++) for (let c=0; c<channels; c++) sum += input.data[((y+ky)*width+x+kx)*channels+c] * weights.data[((f*kh+ky)*kw+kx)*channels+c]
    output.data[(y*ow+x)*filters+f] = sum
  }
  return output
}

export function conv2dBackward(input: TensorValue, weights: TensorValue, bias: TensorValue, upstream: TensorValue): [TensorValue, TensorValue, TensorValue] {
  const { width, channels, filters, kh, kw, oh, ow } = convolutionShape(input, weights, bias)
  if (upstream.shape.join(',') !== [oh,ow,filters].join(',')) throw new Error('Convolution gradient has the wrong shape.')
  const dx = zeros(input.shape), dw = zeros(weights.shape), db = zeros(bias.shape)
  for (let y=0; y<oh; y++) for (let x=0; x<ow; x++) for (let f=0; f<filters; f++) {
    const gradient = upstream.data[(y*ow+x)*filters+f]
    db.data[f] += gradient
    for (let ky=0; ky<kh; ky++) for (let kx=0; kx<kw; kx++) for (let c=0; c<channels; c++) {
      const ii = ((y+ky)*width+x+kx)*channels+c, wi = ((f*kh+ky)*kw+kx)*channels+c
      dx.data[ii] += weights.data[wi] * gradient
      dw.data[wi] += input.data[ii] * gradient
    }
  }
  return [dx,dw,db]
}

export function avgpool2d(input: TensorValue): TensorValue {
  const [height,width,channels] = input.shape
  if (input.shape.length !== 3 || height<2 || width<2) throw new Error('Average pooling expects H×W×C input with H and W at least 2.')
  const oh=Math.floor(height/2),ow=Math.floor(width/2)
  const output = zeros([oh,ow,channels])
  for (let y=0;y<oh;y++) for (let x=0;x<ow;x++) for (let c=0;c<channels;c++) {
    let sum=0
    for (let dy=0;dy<2;dy++) for (let dx=0;dx<2;dx++) sum+=input.data[((2*y+dy)*width+2*x+dx)*channels+c]
    output.data[(y*ow+x)*channels+c]=sum/4
  }
  return output
}

export function avgpool2dBackward(input: TensorValue, upstream: TensorValue): TensorValue {
  const [height,width,channels]=input.shape
  const oh=Math.floor(height/2),ow=Math.floor(width/2)
  if (input.shape.length !== 3 || height<2 || width<2 || upstream.shape.join(',')!==[oh,ow,channels].join(',')) throw new Error('Average-pooling gradient has the wrong shape.')
  const output=zeros(input.shape)
  for(let y=0;y<oh;y++) for(let x=0;x<ow;x++) for(let c=0;c<channels;c++) for(let dy=0;dy<2;dy++) for(let dx=0;dx<2;dx++) output.data[((2*y+dy)*width+2*x+dx)*channels+c]=upstream.data[(y*ow+x)*channels+c]/4
  return output
}
