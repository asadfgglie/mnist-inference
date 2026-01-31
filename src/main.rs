use mnist_inference::*;

fn main() {
    let a = NdArray::new_like(Vec::from_iter((1..28).into_iter()), vec![3, 3, 3]);
    assert_eq!(a.into_iter().map(|x| *x).collect::<Vec<_>>(), Vec::from_iter((1..28).into_iter()));
}