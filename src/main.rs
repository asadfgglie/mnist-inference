use image::ImageReader;
use memmap2::MmapOptions;
use ndarray::ops::arithmetic::relu;
use ndarray::{index, matmul, NdArray, NdArrayLike, Scalar};
use num::Zero;
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::ops::{Add, Mul};

struct Linear<T> {
    weight: NdArray<T>,
    bias: Option<NdArray<T>>,
}

struct SimpleNN {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

trait Forward<T> {
    fn forward(&self, input: & impl NdArrayLike<T>) -> NdArray<T>;
}

impl <T> Forward<T> for Linear<T>
where T: Add<Output=T> + Clone + Mul<Output=T> + Zero {
    fn forward(&self, input: &impl NdArrayLike<T>) -> NdArray<T> {
        // input: (batch_size, in_features)
        let ret: NdArray<T> = matmul(input, &self.weight.transpose(0, 1).unwrap());
        match &self.bias {
            Some(bias) => ret + bias,
            None => ret,
        }
    }
}

impl Forward<f32> for SimpleNN {
    fn forward(&self, input: &impl NdArrayLike<f32>) -> NdArray<f32> {
        let input = self.fc1.forward(input);
        let input = relu(&input);
        self.fc2.forward(&input)
    }
}

fn main() {
    let file = File::open("./mnist_model/model.safetensors").unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();

    let fc1_weight: NdArray<f32> = tensors.tensor("fc1.weight").unwrap().try_into().unwrap();
    let fc1_bias: NdArray<f32> = tensors.tensor("fc1.bias").unwrap().try_into().unwrap();
    let fc2_weight: NdArray<f32> = tensors.tensor("fc2.weight").unwrap().try_into().unwrap();
    let fc2_bias: NdArray<f32> = tensors.tensor("fc2.bias").unwrap().try_into().unwrap();

    let simple_nn = SimpleNN {
        fc1: Linear { weight: fc1_weight, bias: Some(fc1_bias) },
        fc2: Linear { weight: fc2_weight, bias: Some(fc2_bias) },
    };

    let img = ImageReader::open("./test_data/2.bmp")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();

    let (width, height) = img.dimensions();
    assert_eq!((width, height), (28, 28), "MNIST image should be 28x28");

    // 將像素值轉換為 f32 並正規化到 [0, 1]
    let img: Vec<f32> = img
        .into_raw()
        .into_iter()
        .map(|pixel| pixel as f32 / 255.0)
        .collect();

    let img = NdArray::new_shape(img, vec![28,28]);
    let img = NdArray::reshape_array(img, index![1, 28*28]).unwrap();

    let ret: NdArray<f32> = simple_nn.forward(&img);

    let mut ret = ret.map(|&x| {fast_math::exp(x)});
    let dim: Scalar<f32> = Scalar(ret.iter().sum());
    ret /= dim;

    println!("{:?}", ret);

    let (argmax, prob) = ret.data()
        .iter()
        .enumerate()
        .scan((0, 0.0), |(max_i, max), (i, &x)| {
            if x > *max {
                *max = x;
                *max_i = i;
            }
            Some((*max_i, *max))
        })
        .last()
        .unwrap();

    println!("this is `{argmax}` with probability `{prob}`");
}