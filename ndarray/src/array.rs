use std::ops::{Index, IndexMut};
use crate::{Cast, NdArrayError, NdArrayIndex, NdArrayLike};
use crate::ops::arithmetic::{compute_broadcast_strides, compute_reshape_strides, compute_stride, validate_contiguous_stride, validate_view};
use crate::scalar::Scalar;

#[derive(Debug, PartialEq)]
pub struct NdArray<T> {
    pub(crate) data: Box<[T]>,
    pub(crate) shape: NdArrayIndex,
    pub(crate) strides: NdArrayIndex,
    pub(crate) base_offset: usize,
}

#[derive(Debug, PartialEq)]
pub struct NdArrayView<'a, T> {
    pub(crate) data: &'a [T],
    pub(crate) shape: NdArrayIndex,
    pub(crate) strides: NdArrayIndex,
    pub(crate) base_offset: usize,
}

impl <T> NdArrayLike<T> for NdArray<T> {
    fn data<'a, 'b: 'a>(&'b self) -> &'a [T] {
        &self.data
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn base_offset(&self) -> usize {
        self.base_offset
    }
}

impl <T> NdArrayLike<T> for &NdArray<T> {
    fn data<'a, 'b: 'a>(&'b self) -> &'a [T] {
        &self.data
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn base_offset(&self) -> usize {
        self.base_offset
    }
}

impl <'a, T> NdArrayLike<T> for NdArrayView<'a, T> {
    fn data<'b, 'c: 'b>(&'c self) -> &'b [T] {
        self.data
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn base_offset(&self) -> usize {
        self.base_offset
    }
}

impl <'a: 'b, 'b, T> NdArrayLike<T> for &'b NdArrayView<'a, T> {
    fn data<'c ,'d: 'c>(&'d self) -> &'c [T] {
        self.data
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn base_offset(&self) -> usize {
        self.base_offset
    }
}

impl <T> NdArray<T> {
    pub fn new_array(data: Box<[T]>, shape: NdArrayIndex, strides: NdArrayIndex, base_offset: usize) -> Self {
        if data.len() != shape.iter().product() {
            panic!("Data length ({}) does not match shape dimensions ({:?}), shape except {} length data.",
                   data.len(), shape, shape.iter().product::<usize>());
        }

        match validate_contiguous_stride(&shape, &strides) {
            Ok(_) => Self {
                data,
                shape,
                strides,
                base_offset,
            },
            Err(_) => match validate_view(&shape, &shape, &strides) {
                Ok(_) => Self {
                    data,
                    shape,
                    strides,
                    base_offset,
                },
                Err(e) => panic!("{:?}", e)
            }
        }
    }
    pub fn new_shape_with_index(data: Vec<T>, shape: NdArrayIndex) -> Self {
        let stride = compute_stride(&shape);

        Self::new_array(
            data.into_boxed_slice(),
            shape,
            stride,
            0
        )
    }
    pub fn new_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new_shape_with_index(
            data,
            shape.into(),
        )
    }
    pub fn new(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::new_shape(data, shape)
    }

    // shape-related op
    pub fn broadcast_array_to(array: Self, shape: NdArrayIndex) -> Self {
        let stride = match compute_broadcast_strides(array.shape(), array.strides(), &shape) {
            Ok(s) => s,
            Err(e) => panic!("{:?}", e)
        };
        Self::new_array(array.data, shape, stride, array.base_offset)
    }
    pub fn permute_array(array: Self, permutation: NdArrayIndex) -> Self {
        if array.shape().len() != permutation.len() {
            panic!(
                "Illegal shape permutation. target permutation {permutation:?}, old shape: {:?}",
                array.shape(),
            )
        }

        for axis in 0..array.shape().len() {
            if !permutation.contains(&axis) {
                panic!(
                    "axis {axis} not found in target permutation {permutation:?}"
                )
            }
        }

        let mut shape: NdArrayIndex = array.shape().into();
        let mut stride: NdArrayIndex = array.strides().into();
        for axis in 0..array.shape().len() {
            shape[axis] = array.shape()[permutation[axis]];
            stride[axis] = array.strides()[permutation[axis]];
        }

        Self::new_array(array.data, shape, stride, array.base_offset)
    }
    pub fn transpose_array(array: Self, axis1: usize, axis2: usize) -> Self {
        let mut permutation: NdArrayIndex = (0..array.shape().len()).collect::<Vec<usize>>().into();
        let tmp = permutation[axis1];
        permutation[axis1] = permutation[axis2];
        permutation[axis2] = tmp;
        Self::permute_array(array, permutation)
    }

    pub fn item(self) -> Result<Scalar<T>, NdArrayError> {
        if self.shape != NdArrayIndex::Dim1([1]) {
            return Err(NdArrayError::InvalidShapeError(format!(
                "shape must be 1 dimension 1 element array, but got {:?}", self.shape
            )))
        }
        let self_info = format!(
            "self.shape: {:?}, self.stride: {:?}, self.data.len: {}",
            self.shape, self.strides, self.data.len()
        );
        Ok(self.data.into_vec().pop().unwrap_or_else(|| panic!("self.data seems empty, but this should not happen. {}", self_info)).into())
    }
}

impl <T: Clone> NdArray<T> {
    pub fn contiguous(self) -> Self {
        if self.is_contiguous() {
            self
        } else {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());
            data.extend(self.into_iter().cloned());
            Self::new_shape_with_index(data, self.shape.clone())
        }
    }
    pub fn contiguous_self(& mut self) {
        if !self.is_contiguous() {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());
            data.extend(self.into_iter().cloned());
            self.data = data.into_boxed_slice();
            self.strides = compute_stride(&self.shape);
        }
    }

    // shape-related op
    pub fn reshape_array(array: Self, shape: NdArrayIndex) -> Self {
        if array.shape().iter().product::<usize>() != shape.iter().product() {
            panic!("Invalid shape ({:?}) does not match data len ({}).", shape, array.data.len());
        }
        match compute_reshape_strides(array.shape(), array.strides(), &shape) {
            Ok(stride) => Self::new_array(array.data, shape, stride, array.base_offset),
            Err(NdArrayError::IncompatibleReshapeError(_)) => {
                Self::reshape_array(array.contiguous(), shape)
            },
            Err(e) => panic!("{:?}", e)
        }

    }
    pub fn squeeze_array(array: Self, axis: usize) -> Self {
        match array.shape().get(axis) {
            Some(v) => {
                if *v != 1 {
                    array
                } else {
                    let mut shape = array.shape().to_vec();
                    shape.remove(axis);
                    Self::reshape_array(array, shape.into())
                }
            },
            None => {
                panic!("Index out of bounds, shape: {:?}, axis: {axis}", array.shape())
            }
        }
    }
    pub fn unsqueeze_array(array: Self, axis: usize) -> Self {
        match axis <= array.shape().len() {
            true => {
                let mut shape = Vec::with_capacity(array.shape().len() + 1);
                for (i, &j) in array.shape().iter().enumerate() {
                    if i == axis {
                        shape.push(1);
                    }
                    shape.push(j);
                }
                if axis == array.shape().len() {
                    shape.push(1);
                }
                Self::reshape_array(array, shape.into())
            },
            false => {
                panic!("Index out of bounds, shape: {:?}, axis: {axis}", array.shape())
            }
        }
    }
}

impl <'a, T> NdArrayView<'a, T> {
    pub fn new<'b: 'a>(array: &'b [T], old_shape: &[usize], shape: NdArrayIndex, strides: NdArrayIndex, offset: usize) -> Self {
        match validate_view(old_shape, &shape, &strides) {
            Ok(_) => (),
            Err(e) => panic!("{:?}", e)
        }

        Self {
            data: array,
            shape,
            strides,
            base_offset: offset,
        }
    }
}

// Indexing
impl <T, Idx> Index<Idx> for NdArray<T>
where Idx: Into<NdArrayIndex> {
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.compute_index(&index.into())]
    }
}

impl <T, Idx> IndexMut<Idx> for NdArray<T>
where Idx: Into<NdArrayIndex> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.data[self.compute_index(&index.into())]
    }
}

impl <'a, T, Idx> Index<Idx> for NdArrayView<'a, T>
where Idx: Into<NdArrayIndex> {
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.compute_index(&index.into())]
    }
}

// T conversions
impl <T: Clone> From<Vec<T>> for NdArray<T> {
    fn from(data: Vec<T>) -> Self {
        NdArray::new(data)
    }
}

impl <T, NT: Clone> Cast<NT> for NdArray<T>
where NT: From<T> {
    type Target = NdArray<NT>;

    fn cast(self) -> Self::Target {
        NdArray::new_array(
            self.data.into_iter().map(|x| x.into()).collect(),
            self.shape,
            self.strides,
            self.base_offset
        )
    }
}

impl <T: Clone> From<NdArrayView<'_, T>> for NdArray<T> {
    fn from(value: NdArrayView<T>) -> Self {
        let mut data = Vec::with_capacity(value.shape.iter().product());
        data.extend(value.into_iter().cloned());
        Self::new_shape_with_index(data, value.shape)
    }
}