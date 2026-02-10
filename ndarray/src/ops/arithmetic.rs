use crate::array::{NdArray, NdArrayView};
use crate::axis::{broadcast_array, broadcast_shapes, compute_index, compute_stride};
use crate::iterator::IndexIterator;
use crate::scalar::Scalar;
use crate::{NdArrayIndex, NdArrayLike};
use crate::{
    assign_op, assign_scalar_op, eco_op_scalar, general_assign_op, general_eco_op_scalar,
    general_no_eco_op_scalar, general_op, no_eco_op_scalar, op, ref_assign_op, ref_eco_op_scalar,
    ref_no_eco_op_scalar, ref_op, ref_view_op, scalar_assign_op, scalar_op,
};
use num::Zero;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

// NdArray math op
// all NdArrayLike op will through broadcast_array() then become a `NdArrayView`
// so if &NdArrayView $op &NdArrayView is implement => NdArray, NdArrayView, NdArraySource
// and their & version also implement by those macros
// note that all op is element wise operation
general_op! {(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
general_assign_op! {
    (+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign),
    (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)
}
general_no_eco_op_scalar! {(%, Rem, rem), (/, Div, div)}
general_eco_op_scalar! {*, Mul, mul}
assign_scalar_op! {(*=, MulAssign, mul_assign), (%=, RemAssign, rem_assign), (/=, DivAssign, div_assign)}

scalar_op! {(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
scalar_assign_op! {
    (+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign),
    (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)
}

pub fn matmul<L, R>(lhs: &impl NdArrayLike<L>, rhs: &impl NdArrayLike<R>) -> NdArray<L>
where
    L: Clone + Mul<Output = L> + Add<Output = L> + Zero,
    R: Into<L> + Clone,
{
    if lhs.shape().len() == 1 && rhs.shape().len() == 1 {
        // vector dot product
        if lhs.shape() != rhs.shape() {
            panic!(
                "Unexcept shape, vector lhs shape len and vector rhs shape len must same. lhs shape: {:?}, rhs shape: {:?}",
                lhs.shape(),
                rhs.shape()
            )
        }

        let mut ret = L::zero();
        for index in lhs.iter_index() {
            let tmp: L = lhs.data()[lhs.compute_index(&index)].clone()
                * rhs.data()[rhs.compute_index(&index)].clone().into();

            ret = ret + tmp;
        }

        NdArray::new(vec![ret])
    } else if lhs.shape().len() == 1 && rhs.shape().len() > 1 {
        let lhs = match lhs.unsqueeze(0) {
            Ok(l) => l,
            Err(e) => panic!("{e:?}"),
        };
        let ret = matmul(&lhs, rhs);
        let axis = ret.shape().len() - 2;
        NdArray::squeeze_array(ret, axis).expect("squeeze this array should not panic")
    } else if lhs.shape().len() > 1 && rhs.shape().len() == 1 {
        let rhs = match rhs.unsqueeze(1) {
            Ok(r) => r,
            Err(e) => panic!("{e:?}"),
        };
        let ret = matmul(lhs, &rhs);
        let axis = ret.shape().len() - 1;
        NdArray::squeeze_array(ret, axis).expect("squeeze this array should not panic")
    } else if lhs.shape().len() > 1 && rhs.shape().len() > 1 {
        let shape_split = |shape: &[usize]| match shape.len() > 2 {
            true => (
                shape[0..shape.len() - 2].into(),
                shape[shape.len() - 2..].into(),
            ),
            false => (
                NdArrayIndex::Dim0([]),
                <&[usize] as Into<NdArrayIndex>>::into(shape),
            ),
        };
        let ((lhs_batch_shape, lhs_data_shape), (rhs_batch_shape, rhs_data_shape)) =
            (shape_split(lhs.shape()), shape_split(rhs.shape()));

        let (m, p) = (lhs_data_shape[0], lhs_data_shape[1]);
        let (pp, n) = (rhs_data_shape[0], rhs_data_shape[1]);
        if p != pp {
            panic!(
                "Mismatch shape. lhs shape: {:?}, rhs shape: {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }

        let broadcast_batch_shape = match broadcast_shapes(&lhs_batch_shape, &rhs_batch_shape) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}"),
        };

        let lhs = match lhs.broadcast_to(broadcast_batch_shape.clone().concat(lhs_data_shape)) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}"),
        };
        let rhs = match rhs.broadcast_to(broadcast_batch_shape.clone().concat(rhs_data_shape)) {
            Ok(x) => x,
            Err(e) => panic!("{e:?}"),
        };
        // lhs: (broadcast_batch_shape, M, P)
        // rhs: (broadcast_batch_shape, P, N)

        let mut data_shape = broadcast_batch_shape.to_vec();
        data_shape.push(m);
        data_shape.push(n);
        let data_shape: NdArrayIndex = data_shape.into();
        let mut data: Vec<L> = vec![L::zero(); data_shape.iter().product()];
        let data_stride: NdArrayIndex = compute_stride(&data_shape);
        let data_offset = 0;

        let mut inner = |batch_size_index: NdArrayIndex| {
            for i in 0..m {
                for j in 0..n {
                    let mut data_index = batch_size_index.to_vec();
                    data_index.push(i);
                    data_index.push(j);

                    let mut tmp_ret = L::zero();

                    for k in 0..p {
                        let mut lhs_index = batch_size_index.to_vec();
                        let mut rhs_index = batch_size_index.to_vec();

                        lhs_index.push(i);
                        lhs_index.push(k);

                        rhs_index.push(k);
                        rhs_index.push(j);

                        let tmp = lhs[lhs_index].clone() * rhs[rhs_index].clone().into();
                        tmp_ret = tmp_ret + tmp;
                    }

                    data[compute_index(&data_index, &data_stride, data_offset)] = tmp_ret;
                }
            }
        };

        if broadcast_batch_shape.is_empty() {
            inner(NdArrayIndex::Dim0([]));
        } else {
            for batch_size_index in
                IndexIterator::iter_shape(&broadcast_batch_shape).expect("this should not happen")
            {
                inner(batch_size_index);
            }
        }

        NdArray::new_array(
            data.into_boxed_slice(),
            data_shape,
            data_stride,
            data_offset,
        )
    } else {
        panic!(
            "Unexcept shape, lhs shape len and rhs shape len must greater than 0. lhs shape: {:?}, rhs shape: {:?}",
            lhs.shape(),
            rhs.shape()
        )
    }
}

pub fn relu<T: Zero + PartialOrd + Clone>(array: &impl NdArrayLike<T>) -> NdArray<T> {
    let shape: NdArrayIndex = array.shape().into();
    let mut data = Vec::with_capacity(shape.iter().product());
    data.extend(array.iter().map(|x| {
        if *x >= T::zero() {
            x.clone()
        } else {
            T::zero()
        }
    }));

    NdArray::new_shape(data, shape)
}
