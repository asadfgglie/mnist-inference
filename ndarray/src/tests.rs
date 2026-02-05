use super::*;

#[test]
fn test_compute_shape_block() {
    let (shape, stride) = ([9, 7, 8, 6, 5, 4, 3, 2], [40320, 720, 5040, 120, 24, 6, 2, 1]);
    let blocks = compute_shape_block(&shape, &stride);
    assert_eq!(blocks.iter().map(|(b, _)| b).product::<usize>(), shape.iter().product());
}

#[test]
fn test_compute_reshape_stride() {
    let (shape, stride) = ([9,             7,        8,                       6,   5,  4,      3, 2],
                                             [40320,         720,      5040,                    120, 24, 6,      2, 1]);
    let new_shape =                [3,      3,     7,        1,     2,     4,         30,      2,  2,  6];
    let new_stride =      index![120960, 40320, 720,      40320, 20160, 5040,      24,      12, 6,  1];
    assert_eq!(compute_reshape_strides(&shape, &stride, &new_shape).unwrap(), new_stride);

    let (shape, stride) = ([3, 3, 7, 4, 2, 4, 30, 2, 2, 6],
                           [120960, 40320, 720, 0, 20160, 5040, 24, 12, 6, 1]);
    let new_shape = [9,   7,   2,2,    8,   30,4,6];
    let new_stride = index![40320, 720, 0, 0, 5040, 24, 6, 1];
    assert_eq!(compute_reshape_strides(&shape, &stride, &new_shape).unwrap(), new_stride);

    let (shape, stride) = (new_shape, new_stride);
    let new_shape = [9,   7,   4,     8,    720];
    let new_stride = index![40320, 720, 0, 5040, 1];
    assert_eq!(compute_reshape_strides(&shape, &stride, &new_shape).unwrap(), new_stride);

    let (shape, stride) = ([3, 3, 7, 4, 2, 4, 30, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 1]);
    let new_shape = [9,   28,  1,8,   60,2];
    let new_stride = index![0, 0, 0, 0, 0, 1];
    assert_eq!(compute_reshape_strides(&shape, &stride, &new_shape).unwrap(), new_stride);
}