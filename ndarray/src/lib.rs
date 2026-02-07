use ndarray_marco::nd_array_index;
use std::cmp::PartialEq;
use std::ops::DerefMut;
use crate::ops::arithmetic::{compute_broadcast_strides, compute_index, compute_reshape_strides, compute_shape_block};

pub use crate::axis::{AxisSlice, slice};
pub use crate::error::NdArrayError;
pub use crate::scalar::Scalar;
pub use crate::array::{NdArray, NdArrayView};
pub use crate::ops::arithmetic::{broadcast_shapes, broadcast_array, matmul};
pub use crate::iterator::{NdArrayIterator, NdArrayDataIndexIterator, NdArrayFastDataIndexIterator, IndexIterator};

nd_array_index!{8}

pub trait Cast<T> {
    type Target;
    fn cast(self) -> Self::Target;
}

pub trait NdArrayLike<T> {
    fn data<'a, 'b: 'a>(&'b self) -> &'a [T];
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn base_offset(&self) -> usize;
    fn is_contiguous(&self) -> bool {
        compute_shape_block(self.shape(), self.strides()).len() == 1
    }
    fn compute_index(&self, indices: &[usize]) -> usize {
        let index = compute_index(indices, self.strides(), self.base_offset());
        match self.data().get(index) {
            Some(_) => (),
            None => {
                panic!("index {indices:?}=>({index}) out of bounds, shape: {:?}, stride: {:?}", self.shape(), self.strides())
            }
        };
        index
    }
    fn to_view<'a, 'b: 'a>(&'b self) -> NdArrayView<'a, T> {
        NdArrayView::new(self.data(), self.shape(), self.shape().into(), self.strides().into(), self.base_offset())
    }
    fn iter_index<'a>(&'a self) -> NdArrayDataIndexIterator<'a>
    where Self: Sized, T: 'a {
        NdArrayDataIndexIterator::new(self)
    }

    // shape-related op
    fn reshape<'a, 'b: 'a>(&'b self, shape: NdArrayIndex) -> Result<NdArrayView<'a, T>, NdArrayError> {
        let stride = compute_reshape_strides(self.shape(), self.strides(), &shape)?;
        Ok(NdArrayView::new(self.data(), self.shape(), shape, stride, self.base_offset()))
    }
    fn squeeze<'a, 'b: 'a>(&'b self, axis: usize) -> Result<NdArrayView<'a, T>, NdArrayError> {
        match self.shape().get(axis) {
            Some(v) => {
                if *v != 1 {
                    Ok(self.to_view())
                } else {
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    self.reshape(shape.into())
                }
            },
            None => {
                Err(NdArrayError::ReshapeError(format!("Index out of bounds, shape: {:?}, axis: {axis}", self.shape())))
            }
        }
    }
    fn unsqueeze<'a, 'b: 'a>(&'b self, axis: usize) -> Result<NdArrayView<'a, T>, NdArrayError> {
        match axis <= self.shape().len() {
            true => {
                let mut shape = Vec::with_capacity(self.shape().len() + 1);
                for (i, &j) in self.shape().iter().enumerate() {
                    if i == axis {
                        shape.push(1);
                    }
                    shape.push(j);
                }
                if axis == self.shape().len() {
                    shape.push(1);
                }
                self.reshape(shape.into())
            },
            false => {
                Err(NdArrayError::ReshapeError(format!("Index out of bounds, shape: {:?}, axis: {axis}", self.shape())))
            }
        }
    }
    fn broadcast_to<'a, 'b: 'a>(&'b self, shape: NdArrayIndex) -> Result<NdArrayView<'a, T>, NdArrayError> {
        let stride = compute_broadcast_strides(self.shape(), self.strides(), &shape)?;
        Ok(NdArrayView::new(self.data(), self.shape(), shape, stride, self.base_offset()))
    }
    fn permute<'a, 'b: 'a>(&'b self, permutation: NdArrayIndex) -> Result<NdArrayView<'a, T>, NdArrayError> {
        if self.shape().len() != permutation.len() {
            return Err(NdArrayError::PermuteError(format!(
                "Illegal shape permutation. target permutation {permutation:?}, old shape: {:?}",
                self.shape(),
            )))
        }

        for axis in 0..self.shape().len() {
            if !permutation.contains(&axis) {
                return Err(NdArrayError::PermuteError(format!(
                    "axis {axis} not found in target permutation {permutation:?}"
                )))
            }
        }

        let mut shape: NdArrayIndex = self.shape().into();
        let mut stride: NdArrayIndex = self.strides().into();
        for axis in 0..self.shape().len() {
            shape[axis] = self.shape()[permutation[axis]];
            stride[axis] = self.strides()[permutation[axis]];
        }
        Ok(NdArrayView::new(self.data(), self.shape(), shape, stride, self.base_offset()))
    }
    fn transpose<'a, 'b: 'a>(&'b self, axis1: usize, axis2: usize) -> Result<NdArrayView<'a, T>, NdArrayError> {
        let mut permutation: NdArrayIndex = (0..self.shape().len()).collect::<Vec<usize>>().into();
        let tmp = permutation[axis1];
        permutation[axis1] = permutation[axis2];
        permutation[axis2] = tmp;
        self.permute(permutation)
    }

    fn slice<'a, 'b: 'a>(&'b self, slices: &[AxisSlice]) -> Result<NdArrayView<'a, T>, NdArrayError> {
        if slices.len() != self.shape().len() {
            let mut s = Vec::with_capacity(self.shape().len());
            s.extend_from_slice(slices);
            for _ in 0..(self.shape().len() - slices.len()) {
                s.push(AxisSlice::All);
            }
            self.slice(&s)
        } else {
            let mut offset = self.base_offset();
            let mut shape = Vec::with_capacity(self.shape().len());
            let mut strides = Vec::with_capacity(self.strides().len());
            for (axis, slice) in slices.iter().enumerate() {
                let mut process_range = |start: usize, end: usize, step: usize, axis: usize| {
                    if start >= self.shape()[axis] || end > self.shape()[axis] {
                        return Err(NdArrayError::SliceError(format!(
                            "Index out of bounds, shape: {:?}, axis: {axis}, start index: {start}, end index: {end}", self.shape()
                        )))
                    }

                    if start >= end {
                        return Err(NdArrayError::SliceError(format!(
                            "Invalid slice, start index: {start}, end index: {end}",
                        )))
                    }

                    if step == 0 {
                        return Err(NdArrayError::SliceError("Invalid slice 0 step".into()))
                    }

                    shape.push((end - start).div_ceil(step));
                    strides.push(self.strides()[axis] * step);
                    offset += start * self.strides()[axis];
                    Ok(())
                };
                match slice {
                    AxisSlice::All => {
                        shape.push(self.shape()[axis]);
                        strides.push(self.strides()[axis]);
                    }
                    AxisSlice::Index { index } => {
                        if index >= &self.shape()[axis] {
                            return Err(NdArrayError::SliceError(format!(
                                "Index out of bounds, shape: {:?}, axis: {axis}, index: {index}", self.shape()
                            )))
                        }

                        offset += index * self.strides()[axis];
                    }
                    AxisSlice::Range { start, end } => {
                        process_range(*start, *end, 1, axis)?;
                    }
                    AxisSlice::RangeFrom { start } => {
                        process_range(*start, self.shape()[axis], 1, axis)?;
                    },
                    AxisSlice::RangeFromStep { start, step } => {
                        process_range(*start, self.shape()[axis], *step, axis)?;
                    },
                    AxisSlice::RangeTo { end } => {
                        process_range(0, *end, 1, axis)?;
                    },
                    AxisSlice::RangeToStep { end, step } => {
                        process_range(0, *end, *step, axis)?;
                    },
                    AxisSlice::RangeStep { start, end, step } => {
                        process_range(*start, *end, *step, axis)?;
                    },
                    AxisSlice::Step { step} => {
                        process_range(0, self.shape()[axis], *step, axis)?;
                    }
                }
            }
            Ok(NdArrayView::new(self.data(), self.shape(), shape.into(), strides.into(), offset))
        }
    }
}

#[cfg(test)]
mod tests;

pub mod array;
pub mod axis;
pub mod error;
pub mod ops;
pub mod scalar;
pub mod iterator;
