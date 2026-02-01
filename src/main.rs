use mnist_inference::*;
fn main() {
    let mut a: NdArray<f32> = NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast();
    a += NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]);
    assert_eq!(a, NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast() * Scalar(2i8));
}