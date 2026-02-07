// NdArray math op
// all NdArrayLike op will through broadcast_array() then become a `NdArrayView`
// so if &NdArrayView $op &NdArrayView is implement => NdArray, NdArrayView, NdArraySource
// and their & version also implement by those macros
// note that all op is element wise operation
use crate::iterator::IndexIterator;
use crate::scalar::Scalar;
use std::cmp::max;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use crate::{NdArrayError, NdArrayIndex, NdArrayLike, NdArrayView};
use crate::{general_op, general_assign_op, general_no_eco_op_scalar, general_eco_op_scalar, assign_scalar_op,
            op, ref_op, ref_view_op, assign_op, ref_assign_op, no_eco_op_scalar, eco_op_scalar, ref_no_eco_op_scalar, ref_eco_op_scalar};
use crate::array::NdArray;

general_op!{(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
general_assign_op!{
    (+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign),
    (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)
}
general_no_eco_op_scalar!{(%, Rem, rem), (/, Div, div)}
general_eco_op_scalar!{*, Mul, mul}
assign_scalar_op!{(*=, MulAssign, mul_assign), (%=, RemAssign, rem_assign), (/=, DivAssign, div_assign)}


/// base on shape, return row-major contiguous stride
pub(crate) fn compute_stride(shape: &[usize]) -> NdArrayIndex {
    shape.iter()
        .rev()
        .scan(1, |acc, x| {
            let tmp = *acc;
            *acc *= x;
            Some(tmp)
        })
        .collect::<Vec<usize>>()
        .into_iter()
        .rev()
        .collect::<Vec<usize>>()
        .into()
}

/// ```rust
/// use ndarray::broadcast_shapes;
/// use ndarray::NdArrayError;
/// use ndarray::index;
///
/// // Example 1: Broadcasting compatible shapes
/// let shape1 = vec![2, 3, 4];
/// let shape2 = vec![3, 1];
/// let result = broadcast_shapes(&shape1, &shape2).unwrap();
/// assert_eq!(result, index![2, 3, 4]);
///
/// // Example 2: Incompatible shapes
/// let shape3 = vec![2, 3];
/// let shape4 = vec![4, 5];
/// let result = broadcast_shapes(&shape3, &shape4);
/// assert!(matches!(result, Err(NdArrayError::BroadcastError(_))));
///
/// // Example 3: Shapes with different lengths
/// let shape5 = vec![2, 3, 4, 5];
/// let shape6 = vec![2, 3, 1, 5];
/// let result = broadcast_shapes(&shape5, &shape6).unwrap();
/// assert_eq!(result, index![2, 3, 4, 5]);
///
/// // Example 4
/// let shape7 = vec![1];
/// let shape8 = vec![4];
/// let result = broadcast_shapes(&shape7, &shape8).unwrap();
/// assert_eq!(result, index![4]);
/// ```
pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<NdArrayIndex, NdArrayError> {
    let longer = if lhs.len() > rhs.len() { lhs } else { rhs };
    let shorter = if lhs.len() > rhs.len() { rhs } else { lhs };

    let mut longer = longer.iter().rev();
    let mut shorter = shorter.iter().rev();

    let mut ret = Vec::with_capacity(longer.len());
    loop {
        let (l, r) = (longer.next(), shorter.next());

        if l.is_none() && r.is_none() {
            break;
        }

        let (l, r) = (match l {
            Some(x) => *x,
            None => 1
        }, match r {
            Some(x) => *x,
            None => 1
        });

        if l == 1 || r == 1 {
            ret.push(max(l, r))
        } else if l == r {
            ret.push(l)
        } else {
            return Err(NdArrayError::BroadcastError(format!(
                "Shapes {:?} and {:?} are incompatible for broadcasting.", lhs, rhs
            )))
        }
    };

    ret.reverse();
    Ok(ret.into())
}


/// return Vec<(block_elements, base_stride)>
pub(crate) fn compute_shape_block(shape: &[usize], stride: &[usize]) -> Vec<(usize, usize)> {
    assert_eq!(shape.len(), stride.len());
    assert_ne!(shape.len(), 0);
    let mut blocks: Vec<(usize, usize)> = Vec::with_capacity(shape.len());
    let mut current_block: Vec<usize> = Vec::with_capacity(shape.len());
    let mut base_stride: usize = 0;
    for (axis, (&d, &s)) in zip(shape.iter(), stride.iter()).enumerate().rev() {
        if current_block.is_empty() {
            if s != 0 {
                base_stride = s;
                current_block.push(d);
            } else {
                // a broadcast axis itself is a total d elements, 0 stride block
                // so contiguous 0 stride => same block with 0 base stride
                base_stride = s;
                current_block.push(d);
            }
        } else if s != 0 {
            let next_axis = axis + 1;
            if s != shape[next_axis] * stride[next_axis] {
                // shape block bound found
                blocks.push((current_block.iter().product(), base_stride));
                current_block.clear();
                base_stride = s;
            }
            current_block.push(d);
        } else if base_stride == s {
            // contiguous 0 strides => same block with 0 base stride
            current_block.push(d);
        } else {
            // s == 0 but base_stride != 0 => meet broadcast axis
            blocks.push((current_block.iter().product(), base_stride));
            current_block.clear();
            base_stride = s;
            current_block.push(d);
        }
    }
    if !current_block.is_empty() {
        blocks.push((current_block.iter().product(), base_stride));
    }
    blocks
}

pub(crate) fn compute_reshape_strides(old_shape: &[usize], old_stride: &[usize], reshape: &[usize]) -> Result<NdArrayIndex, NdArrayError> {
    if old_shape.is_empty() {
        return Err(NdArrayError::InvalidShapeError("shape is empty!".into()))
    }

    if old_shape.len() != old_stride.len() {
        return Err(NdArrayError::InvalidShapeError(format!(
            "Shape len and stride len doesn't match. shape len: {}, stride len: {}",
            old_shape.len(), old_stride.len()
        )))
    }

    if old_shape.iter().product::<usize>() != reshape.iter().product() {
        return Err(NdArrayError::ReshapeError(format!(
            "Unable turn shape {:?} ({}) into new shape {:?} ({}) since element number mismatch.",
            old_shape, old_shape.iter().product::<usize>(), reshape, reshape.iter().product::<usize>()
        )))
    }

    if old_shape.iter().product::<usize>() == 0 {
        return Ok(reshape.into())
    }

    let err = Err(NdArrayError::IncompatibleReshapeError(format!(
        "old shape {:?} and shape {:?} isn't compatible. consider contiguous array then reshape again.",
        old_shape, reshape
    )));
    let mut reshape_iter = reshape.iter().rev().peekable();
    let mut stride = Vec::with_capacity(reshape.len());
    for (block_elements, base_stride) in compute_shape_block(old_shape, old_stride) {
        stride.push(base_stride);

        let mut current_elements = match reshape_iter.next() {
            Some(&x) => x,
            None => return err
        };
        let mut next_d = current_elements;

        while current_elements <= block_elements {
            let d = match reshape_iter.peek() {
                Some(&&x) => x,
                None => {
                    if current_elements == block_elements {
                        break;
                    }
                    return err
                }
            };
            current_elements *= d;

            if current_elements <= block_elements {
                stride.push(next_d * stride.last().expect("this shouldn't happen"));
                next_d = d;
                reshape_iter.next().expect("this shouldn't happen");
            }
        }
    }

    stride.reverse();
    Ok(stride.into())
}

pub(crate) fn compute_broadcast_strides(old_shape: &[usize], old_stride: &[usize], broadcast_shape: &[usize]) -> Result<NdArrayIndex, NdArrayError> {
    let mut strides: Vec<usize> = Vec::with_capacity(broadcast_shape.len());

    let mut old_shape_iter = old_shape.iter().rev();
    let mut broadcast_shape_iter = broadcast_shape.iter().rev();
    let mut old_stride_iter = old_stride.iter().rev();

    loop {
        let (o, b) = (old_shape_iter.next(), broadcast_shape_iter.next());

        if o.is_none() && b.is_none() {
            break;
        }

        let (o, b) = (match o {
            Some(x) => *x,
            None => 0 // hidden broadcast dimension, you can treat as unsqueeze axis
        }, match b {
            Some(x) => *x,
            None => return Err(NdArrayError::BroadcastError(format!(
                "broadcast_shape shorter than old_array.shape, broadcast_shape len: {}, \
                                old_array.shape len: {}",
                broadcast_shape.len(),
                old_shape.len()
            )))
        });

        if (o == 1 || o == 0) && b > o { // this is the broadcast dimension,
            // so stride is always 0
            strides.push(0)
        } else {
            strides.push(*old_stride_iter.next().expect(
                "Unable to get old stride. old_stride_iter.next() is None. This should never happen."
            ))
        }
    }

    strides.reverse();
    Ok(strides.into())
}


/// Broadcasts the shapes of two `NdArray` objects to ensure compatibility for operations
/// that require aligned shapes (e.g., element-wise operations).
///
/// This function computes the broadcasted shapes and strides for each input array,
/// ensuring that both arrays can align their dimensions according to broadcasting rules.
///
/// # T Parameters
/// - `'a`: Lifetime of the first input array (`rhs`).
/// - `'b`: Lifetime of the second input array (`lhs`).
/// - `L`: Data T of the elements in the first input array (`rhs`).
/// - `R`: Data T of the elements in the second input array (`lhs`).
///
/// # Arguments
/// - `rhs`: A reference to the first `NdArray` object.
/// - `lhs`: A reference to the second `NdArray` object.
///
/// # Returns
/// - `Ok((NdArrayView<'a, L>, NdArrayView<'b, R>))`: A tuple containing views of the broadcasted
///   forms of the input arrays. Each view has the broadcasted shape and appropriate strides.
/// - `Err(NdArrayError)`: An error if the shapes of the input arrays do not align for broadcasting.
///
/// # Broadcasting Rules
/// - Broadcasting follows the standard numpy-like rules:
///   - Dimensions with size `1` can be broadcasted to match a dimension of the other array.
///   - Dimensions of the input arrays must either be equal or one of them must be `1`.
///   - If the shapes of two arrays are not compatible under these rules, an error is returned.
///
/// # Panics
/// This function will panic under the following conditions:
/// - If the computed broadcast shape is shorter than the original shape of an input array.
/// - If an unexpected error occurs during the computation of strides.
///
/// # Example
/// ```
/// use ndarray::{broadcast_array, NdArray, NdArrayLike};
///
/// let array1 = NdArray::new_shape(vec![0.0, 1.0, 2.0], vec![1, 3, 1]);
/// let array2 = NdArray::new_shape(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 1, 3]);
///
/// match broadcast_array(&array1, &array2) {
///     Ok((view1, view2)) => {
///         assert_eq!(view1.shape(), &vec![2, 3, 3]);
///         assert_eq!(view2.shape(), &vec![2, 3, 3]);
///     }
///     Err(e) => {
///         panic!("{:?}", e)
///     }
/// }
/// ```
///
/// # Implementation Notes
/// - The computation of strides is done in reverse order to match the layout of high-dimensional
///   arrays.
/// - Strides for broadcasted dimensions are set to `0` to ensure proper indexing behavior.
/// - The function clones the computed broadcast shape to ensure immutability and avoid accidental
///   modifications during stride computation.
pub fn broadcast_array<'a, 'b, 'c: 'a, 'd: 'b, L, R>(lhs: &'c impl NdArrayLike<L>, rhs: &'d impl NdArrayLike<R>)
                                                     -> Result<(NdArrayView<'a, L>, NdArrayView<'b, R>), NdArrayError> {
    match broadcast_shapes(lhs.shape(), rhs.shape()) {
        Ok(broadcast_shape) => {
            let rhs_shape = broadcast_shape.clone();
            let lhs_shape = broadcast_shape;

            let lhs_strides = compute_broadcast_strides(lhs.shape(), lhs.strides(), &lhs_shape)?;
            let rhs_strides = compute_broadcast_strides(rhs.shape(), rhs.strides(), &rhs_shape)?;

            Ok((NdArrayView::new(lhs.data(), lhs.shape(), lhs_shape, lhs_strides, lhs.base_offset()),
                NdArrayView::new(rhs.data(), rhs.shape(), rhs_shape, rhs_strides, lhs.base_offset())))
        }
        Err(e) => Err(e)
    }
}

pub(crate) fn compute_index(indices: &[usize], strides: &[usize], base_offset: usize) -> usize {
    if indices.len() != strides.len() {
        panic!(
            "Indices length ({}) does not match NdArray dimensions ({})",
            indices.len(),
            strides.len()
        );
    }

    indices.iter()
        .enumerate()
        .fold(0, |acc, (axis, &x)| acc + x * strides[axis]) + base_offset
}

pub(crate) fn validate_view(old_shape: &[usize], view_shape: &[usize], view_stride: &[usize]) -> Result<(), NdArrayError> {
    if view_shape.len() != view_stride.len() {
        Err(NdArrayError::InvalidStridesError(format!(
            "view_shape.len() ({}) != view_stride.len() ({})", view_shape.len(), view_stride.len()
        )))
    } else {
        let mut old_shape_iter = old_shape.iter().rev();
        let mut view_shape_iter = view_shape.iter().rev();
        let mut view_stride_iter = view_stride.iter().rev();

        loop {
            let (o, v, s) =
                (old_shape_iter.next(), view_shape_iter.next(), view_stride_iter.next());
            if o.is_none() && v.is_none() && s.is_none() {
                break;
            }

            let (o, v, s) = (match o {
                Some(x) => *x,
                None => 1,
            }, match v {
                Some(x) => *x,
                None => return Err(NdArrayError::InvalidStridesError(format!(
                    "view_shape shorter than old_shape, view_shape len: {}, \
                      old_shape len: {}",
                    view_shape.len(),
                    old_shape.len()
                )))
            },
                             match s {
                                 Some(x) => *x,
                                 None => return Err(NdArrayError::InvalidStridesError(format!(
                                     "view_stride shorter than old_shape, view_stride len: {}, \
                      old_shape len: {}",
                                     view_stride.len(),
                                     old_shape.len()
                                 )))
                             });

            if v != o && o != 1 {
                return Err(NdArrayError::BroadcastError(format!(
                    "Cannot create a view with shape {:?} from an array of shape {:?}.",
                    view_shape, old_shape
                )))
            } else if v > o && o == 1 && s != 0 {
                return Err(NdArrayError::InvalidStridesError(format!(
                    "Strides {:?} are invalid for a view of shape {:?}, old shape: {:?}. \
                    Broadcast axis stride must be zero!",
                    view_stride, view_shape, old_shape
                )))
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_contiguous_stride(shape: &[usize], strides: &[usize]) -> Result<(), NdArrayError> {
    let contiguous_stride = compute_stride(shape);
    if strides == contiguous_stride {
        Ok(())
    } else {
        Err(NdArrayError::InvalidStridesError(
            format!("Except stride {:?}, got {:?}", contiguous_stride, strides)
        ))
    }
}

pub fn matmul<L: Clone + Mul<Output=L> + Add<Output=L>, R: Into<L> + Clone>(lhs: & impl NdArrayLike<L>, rhs: & impl NdArrayLike<R>) -> NdArray<L> {
    if lhs.shape().len() == 1 && rhs.shape().len() == 1 { // vector dot product
        if lhs.shape() != rhs.shape() {
            panic!(
                "Unexcept shape, vector lhs shape len and vector rhs shape len must same. lhs shape: {:?}, rhs shape: {:?}",
                lhs.shape(), rhs.shape()
            )
        }

        let mut ret: Vec<L> = Vec::with_capacity(1);
        for index in lhs.iter_index() {
            let tmp: L = lhs.data()[lhs.compute_index(&index)].clone() * rhs.data()[rhs.compute_index(&index)].clone().into();

            match ret.is_empty() {
                true => ret.push(tmp),
                false => {
                    let v = ret.pop().unwrap();
                    ret.push(v + tmp);
                }
            }
        }

        assert_eq!(ret.len(), 1);
        NdArray::new(ret)
    } else if lhs.shape().len() == 1 && rhs.shape().len() > 1 {
        let lhs = match lhs.unsqueeze(0) {
            Ok(l) => l,
            Err(e) => panic!("{e:?}")
        };
        let ret = matmul(&lhs, rhs);
        let axis = ret.shape().len() - 2;
        NdArray::squeeze_array(ret, axis)
    } else if lhs.shape().len() > 1 && rhs.shape().len() == 1 {
        let rhs = match rhs.unsqueeze(1) {
            Ok(r) => r,
            Err(e) => panic!("{e:?}")
        };
        let ret = matmul(lhs, &rhs);
        let axis = ret.shape().len() - 1;
        NdArray::squeeze_array(ret, axis)
    } else if lhs.shape().len() > 1 && rhs.shape().len() > 1 {
        let shape_split = |shape: &[usize]| {
            match shape.len() > 2 {
                true => (shape[0..shape.len() - 2].into(), shape[shape.len() - 2..].into()),
                false => (NdArrayIndex::Dim0([]), <&[usize] as Into<NdArrayIndex>>::into(shape))
            }
        };
        let ((lhs_batch_shape, lhs_data_shape),
            (rhs_batch_shape, rhs_data_shape)) = (
            shape_split(lhs.shape()), shape_split(rhs.shape()),
        );

        let (m, p) = (lhs_data_shape[0], lhs_data_shape[1]);
        let (pp, n) = (rhs_data_shape[0], rhs_data_shape[1]);
        if p != pp {
            panic!("Mismatch shape. lhs shape: {:?}, rhs shape: {:?}", lhs.shape(), rhs.shape());
        }

        let broadcast_batch_shape = match broadcast_shapes(&lhs_batch_shape, &rhs_batch_shape) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}")
        };

        let lhs = match lhs.broadcast_to(broadcast_batch_shape.clone().concat(lhs_data_shape)) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}")
        };
        let rhs = match rhs.broadcast_to(broadcast_batch_shape.clone().concat(rhs_data_shape)) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}")
        };
        // lhs: (broadcast_batch_shape, M, P)
        // rhs: (broadcast_batch_shape, P, N)

        let mut data_shape = broadcast_batch_shape.to_vec();
        data_shape.push(m);
        data_shape.push(n);
        let data_shape: NdArrayIndex = data_shape.into();
        let mut data: Vec<L> = Vec::with_capacity(data_shape.iter().product());
        let data_stride: NdArrayIndex = compute_stride(&data_shape);
        let data_offset = 0;

        for batch_size_index in IndexIterator::iter_shape(&broadcast_batch_shape) {
            for i in 0..m {
                for j in 0..n {
                    let mut data_index = batch_size_index.to_vec();
                    data_index.push(i);
                    data_index.push(j);

                    let mut tmp_ret: Vec<L> = Vec::with_capacity(1);

                    for k in 0..p {
                        let mut lhs_index = batch_size_index.to_vec();
                        let mut rhs_index = batch_size_index.to_vec();

                        lhs_index.push(i);
                        lhs_index.push(k);

                        rhs_index.push(k);
                        rhs_index.push(j);

                        let tmp = lhs[lhs_index].clone() * rhs[rhs_index].clone().into();
                        match tmp_ret.is_empty() {
                            true => tmp_ret.push(tmp),
                            false => {
                                let v = tmp_ret.pop().unwrap();
                                tmp_ret.push(v + tmp);
                            }
                        }
                    }

                    assert_eq!(tmp_ret.len(), 1);
                    data[compute_index(&data_index, &data_stride, data_offset)] = tmp_ret.pop().unwrap();
                }
            }
        }

        NdArray::new_array(data.into_boxed_slice(), data_shape, data_stride, data_offset)
    } else {
        panic!(
            "Unexcept shape, lhs shape len and rhs shape len must greater than 0. lhs shape: {:?}, rhs shape: {:?}",
            lhs.shape(), rhs.shape()
        )
    }
}