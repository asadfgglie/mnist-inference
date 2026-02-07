use ndarray::*;
use ndarray::array::NdArray;

fn main() {
    let _ = NdArray::new_shape(vec![0;9], vec![3, 3]);
    let _: NdArrayIndex = index![0,1];
    let _ = [0;0];
    println!("{}", size_of::<[usize;0]>())
}