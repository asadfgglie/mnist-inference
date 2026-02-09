use crate::axis::{
    compute_broadcast_strides, compute_reshape_strides, compute_stride, validate_view,
};
use crate::scalar::Scalar;
use crate::{Cast, HasDtype, NdArrayError, NdArrayIndex, NdArrayLike};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, View};
use std::borrow::Cow;
use std::ops::{Index, IndexMut};

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

impl<T> NdArrayLike<T> for NdArray<T> {
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

impl<T> NdArrayLike<T> for &NdArray<T> {
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

impl<'a, T> NdArrayLike<T> for NdArrayView<'a, T> {
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

impl<'a: 'b, 'b, T> NdArrayLike<T> for &'b NdArrayView<'a, T> {
    fn data<'c, 'd: 'c>(&'d self) -> &'c [T] {
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

/// Shape-related api will do the same thing as `NdArrayLike` trait when success
/// and then return `NdArray`, not `NdArrayView`.
impl<T> NdArray<T> {
    pub fn new_array(
        data: Box<[T]>,
        shape: NdArrayIndex,
        strides: NdArrayIndex,
        base_offset: usize,
    ) -> Self {
        if data.len() != shape.iter().product::<usize>() {
            panic!(
                "Data length ({}) does not match shape dimensions ({:?}), shape except {} length data.",
                data.len(),
                shape,
                shape.iter().product::<usize>()
            );
        }

        match validate_view(data.len(), base_offset, &shape, &strides) {
            Ok(_) => Self {
                data,
                shape,
                strides,
                base_offset,
            },
            Err(e) => panic!("{:?}", e),
        }
    }
    pub fn new_shape_with_index(data: Vec<T>, shape: NdArrayIndex) -> Self {
        let stride = compute_stride(&shape);

        Self::new_array(data.into_boxed_slice(), shape, stride, 0)
    }
    pub fn new_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new_shape_with_index(data, shape.into())
    }
    pub fn new(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::new_shape(data, shape)
    }

    // shape-related op
    pub fn broadcast_array_to(array: Self, shape: NdArrayIndex) -> Result<Self, NdArrayError> {
        let stride = compute_broadcast_strides(array.shape(), array.strides(), &shape)?;
        Ok(Self::new_array(
            array.data,
            shape,
            stride,
            array.base_offset,
        ))
    }
    pub fn permute_array(array: Self, permutation: NdArrayIndex) -> Result<Self, NdArrayError> {
        if array.shape().len() != permutation.len() {
            return Err(NdArrayError::PermuteError(format!(
                "Illegal shape permutation. target permutation {permutation:?}, old shape: {:?}",
                array.shape(),
            )));
        }

        for axis in 0..array.shape().len() {
            if !permutation.contains(&axis) {
                return Err(NdArrayError::PermuteError(format!(
                    "axis {axis} not found in target permutation {permutation:?}"
                )));
            }
        }

        let mut shape: NdArrayIndex = array.shape().into();
        let mut stride: NdArrayIndex = array.strides().into();
        for axis in 0..array.shape().len() {
            shape[axis] = array.shape()[permutation[axis]];
            stride[axis] = array.strides()[permutation[axis]];
        }

        Ok(Self::new_array(
            array.data,
            shape,
            stride,
            array.base_offset,
        ))
    }
    pub fn transpose_array(array: Self, axis1: usize, axis2: usize) -> Result<Self, NdArrayError> {
        let mut permutation: NdArrayIndex = (0..array.shape().len()).collect::<Vec<usize>>().into();
        let tmp = permutation[axis1];
        permutation[axis1] = permutation[axis2];
        permutation[axis2] = tmp;
        Self::permute_array(array, permutation)
    }

    pub fn map_self(&mut self, f: impl Fn(&T) -> T) {
        for indices in self.iter_index().collect::<Vec<_>>() {
            self[indices] = f(&self[indices.clone()]);
        }
    }

    pub fn item(self) -> Result<Scalar<T>, NdArrayError> {
        if self.shape != NdArrayIndex::Dim1([1]) {
            return Err(NdArrayError::InvalidShapeError(format!(
                "shape must be 1 dimension 1 element array, but got {:?}",
                self.shape
            )));
        }
        let self_info = format!(
            "self.shape: {:?}, self.stride: {:?}, self.data.len: {}",
            self.shape,
            self.strides,
            self.data.len()
        );
        Ok(self
            .data
            .into_vec()
            .pop()
            .unwrap_or_else(|| panic!("this should not happen. {}", self_info))
            .into())
    }
}

impl<T: Clone> NdArray<T> {
    pub fn contiguous(self) -> Self {
        if self.is_contiguous() {
            self
        } else {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());
            data.extend(self.into_iter().cloned());
            Self::new_shape_with_index(data, self.shape.clone())
        }
    }
    pub fn contiguous_self(&mut self) {
        if !self.is_contiguous() {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());
            data.extend(self.into_iter().cloned());
            self.data = data.into_boxed_slice();
            self.strides = compute_stride(&self.shape);
        }
    }

    // shape-related op
    pub fn reshape_array(array: Self, shape: NdArrayIndex) -> Result<Self, NdArrayError> {
        match compute_reshape_strides(array.shape(), array.strides(), &shape) {
            Ok(stride) => Ok(Self::new_array(
                array.data,
                shape,
                stride,
                array.base_offset,
            )),
            Err(NdArrayError::IncompatibleReshapeError(_)) => {
                Self::reshape_array(array.contiguous(), shape)
            }
            Err(e) => Err(e),
        }
    }
    pub fn squeeze_array(array: Self, axis: usize) -> Result<Self, NdArrayError> {
        match array.shape().get(axis) {
            Some(v) => {
                if *v != 1 {
                    Ok(array)
                } else {
                    let mut shape = array.shape().to_vec();
                    shape.remove(axis);
                    Self::reshape_array(array, shape.into())
                }
            }
            None => Err(NdArrayError::ReshapeError(format!(
                "Index out of bounds, shape: {:?}, axis: {axis}",
                array.shape()
            ))),
        }
    }
    pub fn unsqueeze_array(array: Self, axis: usize) -> Result<Self, NdArrayError> {
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
            }
            false => Err(NdArrayError::ReshapeError(format!(
                "Index out of bounds, shape: {:?}, axis: {axis}",
                array.shape()
            ))),
        }
    }
}

impl<'a, T> NdArrayView<'a, T> {
    pub fn new<'b: 'a>(
        array: &'b [T],
        shape: NdArrayIndex,
        strides: NdArrayIndex,
        offset: usize,
    ) -> Self {
        match validate_view(array.len(), offset, &shape, &strides) {
            Ok(_) => (),
            Err(e) => panic!("{:?}", e),
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
impl<T, Idx> Index<Idx> for NdArray<T>
where
    Idx: Into<NdArrayIndex>,
{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.compute_index(&index.into())]
    }
}

impl<T, Idx> IndexMut<Idx> for NdArray<T>
where
    Idx: Into<NdArrayIndex>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.data[self.compute_index(&index.into())]
    }
}

impl<'a, T, Idx> Index<Idx> for NdArrayView<'a, T>
where
    Idx: Into<NdArrayIndex>,
{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.compute_index(&index.into())]
    }
}

// T conversions
impl<T: Clone> From<Vec<T>> for NdArray<T> {
    fn from(data: Vec<T>) -> Self {
        NdArray::new(data)
    }
}

impl<T, NT: Clone> Cast<NT> for NdArray<T>
where
    NT: From<T>,
{
    type Target = NdArray<NT>;

    fn cast(self) -> Self::Target {
        NdArray::new_array(
            self.data.into_iter().map(|x| x.into()).collect(),
            self.shape,
            self.strides,
            self.base_offset,
        )
    }
}

impl<T: Clone> From<NdArrayView<'_, T>> for NdArray<T> {
    fn from(value: NdArrayView<T>) -> Self {
        let mut data = Vec::with_capacity(value.shape.iter().product());
        data.extend(value.into_iter().cloned());
        Self::new_shape_with_index(data, value.shape)
    }
}

impl<'a, T> TryFrom<TensorView<'a>> for NdArrayView<'a, T>
where
    T: HasDtype,
{
    type Error = NdArrayError;

    fn try_from(value: TensorView<'a>) -> Result<Self, Self::Error> {
        if value.dtype() != T::DTYPE {
            return Err(NdArrayError::DtypeMismatch(format!(
                "Tensor dtype mismatch, expected {:?}, got {:?}",
                T::DTYPE,
                value.dtype()
            )));
        }

        let shape: NdArrayIndex = value.shape().into();
        let elements: usize = shape.iter().product();

        let expected_bytes = elements * size_of::<T>();
        if value.data().len() != expected_bytes {
            return Err(NdArrayError::InvalidBufferSize(format!(
                "Tensor size mismatch, expected {:?}, got {:?}",
                expected_bytes,
                value.data().len()
            )));
        }

        let ptr = value.data().as_ptr();
        if ptr.align_offset(align_of::<T>()) != 0 {
            return Err(NdArrayError::Misaligned(format!(
                "Tensor alignment mismatch, expected {:?}, got {:?}",
                align_of::<T>(),
                ptr.align_offset(align_of::<T>())
            )));
        }

        let strides = compute_stride(&shape);

        let typed_slice: &'a [T] = unsafe { std::slice::from_raw_parts(ptr as *const T, elements) };

        Ok(NdArrayView::new(typed_slice, shape, strides, 0))
    }
}

impl<'a, T> TryFrom<TensorView<'a>> for NdArray<T>
where
    T: HasDtype + Clone,
{
    type Error = NdArrayError;

    fn try_from(value: TensorView<'a>) -> Result<Self, Self::Error> {
        Ok(NdArrayView::try_from(value)?.into())
    }
}

impl<T> View for &NdArray<T>
where
    T: HasDtype,
{
    fn dtype(&self) -> Dtype {
        T::DTYPE
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        if !self.is_contiguous() {
            panic!("Non-contiguous NdArray cannot be serialized to safetensors");
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, size_of_val(&self.data))
        };

        Cow::Borrowed(bytes)
    }

    fn data_len(&self) -> usize {
        View::data(self).len()
    }
}

impl<T> View for &NdArrayView<'_, T>
where
    T: HasDtype,
{
    fn dtype(&self) -> Dtype {
        T::DTYPE
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        if !self.is_contiguous() {
            panic!("Non-contiguous NdArrayView cannot be serialized to safetensors");
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, size_of_val(self.data))
        };

        Cow::Borrowed(bytes)
    }

    fn data_len(&self) -> usize {
        View::data(self).len()
    }
}
