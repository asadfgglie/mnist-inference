use super::*;

#[test]
fn test_compute_shape_block() {
    let (shape, stride) = ([9, 7, 8, 6, 5, 4, 3, 2], [40320, 720, 5040, 120, 24, 6, 2, 1]);
    let blocks = compute_shape_block(&shape, &stride);
    assert_eq!(blocks.iter().map(|(b, _)| b).product::<usize>(), shape.iter().product(), "miss axis");
}

#[test]
fn test_compute_reshape_stride() {
    let (shape, stride) = ([9,             7,        8,                       6,   5,  4,      3, 2],
                                             [40320,         720,      5040,                    120, 24, 6,      2, 1]);
    let new_shape =                [3,      3,     7,        1,     2,     4,         30,      2,  2,  6];
    let new_stride =      index![120960, 40320, 720,      40320, 20160, 5040,      24,      12, 6,  1];
    assert_eq!(compute_reshape_strides(&shape, &stride, &new_shape).unwrap(), new_stride)
}