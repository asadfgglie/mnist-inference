use std::marker::PhantomData;
use crate::array::{NdArray, NdArrayView};
use crate::NdArrayLike;
use crate::NdArrayIndex;
use crate::axis::compute_index;


pub struct NdArrayIterator<'a, T: NdArrayLike<DT>, DT> {
    data: &'a T,
    index_iter: NdArrayFastDataIndexIterator<'a>,
    _marker: PhantomData<DT>,
}

/// Iter logical index like `vec![1, 2, 0]` by array shape.
///
/// It will check by itself to ensure the index is valid.
///
/// Avoid using this for a big array. This will cause lots of allocated
pub struct NdArrayDataIndexIterator<'a> {
    data_len: usize,
    data_strides: &'a [usize],
    data_offset: usize,
    iter: IndexIterator<'a>
}

/// Use for NdArrayIterator prevents lots of Vec allocated.
///
/// This yields a valid physical index of linear memory data, not a logical index like `vec![1, 2, 0]`
pub struct NdArrayFastDataIndexIterator<'a> {
    data_len: usize,
    data_strides: &'a [usize],
    data_shape: &'a [usize],
    data_offset: usize,
    index: NdArrayIndex,
    axis_counter: usize,
    has_done: bool,
}

/// Iter logical index like `vec![1, 2, 0]` by shape.
///
/// Avoid using this for a big shape. This will cause lots of allocated.
pub struct IndexIterator<'a> {
    data_shape: &'a [usize],
    index: NdArrayIndex,
    axis_counter: usize,
    has_done: bool,
}

impl <'a, T: NdArrayLike<DT>, DT> NdArrayIterator<'a, T, DT> {
    pub fn new<'b: 'a>(array: &'b T) -> Self {
        Self {
            data: array,
            index_iter: NdArrayFastDataIndexIterator::iter_index(array.shape(), array.strides(), array.base_offset()),
            _marker: PhantomData,
        }
    }
}

impl <'a> NdArrayDataIndexIterator<'a> {
    pub fn new<'b: 'a, T>(array: &'b impl NdArrayLike<T>) -> Self {
        Self {
            data_len: array.data().len(),
            data_strides: array.strides(),
            iter: IndexIterator::iter_shape(array.shape()),
            data_offset: array.base_offset(),
        }
    }
}

impl <'a> NdArrayFastDataIndexIterator<'a> {
    pub fn iter_index<'b: 'a>(shape: &'b [usize], strides: &'b [usize], offset: usize) -> Self {
        Self {
            data_len: shape.iter().product(),
            data_strides: strides,
            data_shape: shape,
            index: NdArrayIndex::zeros(shape.len()),
            axis_counter: shape.len() - 1,
            has_done: false,
            data_offset: offset,
        }
    }
}

impl <'a> IndexIterator<'a> {
    pub fn iter_shape<'b: 'a>(shape: &'b [usize]) -> Self {
        Self {
            index: NdArrayIndex::zeros(shape.len()),
            axis_counter: shape.len() - 1,
            has_done: false,
            data_shape: shape,
        }
    }
}


// Iterators
impl <'a, T: NdArrayLike<DT>, DT: 'a> Iterator for NdArrayIterator<'a, T, DT> {
    type Item = &'a DT;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            None => None,
            Some(index) => Some(&self.data.data()[index]),
        }
    }
}

impl <'a> Iterator for NdArrayDataIndexIterator<'a> {
    type Item = NdArrayIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some(index) => {
                let i = compute_index(&index, self.data_strides, self.data_offset);
                if i >= self.data_len {
                    panic!(
                        "Index out of bounds. Index: {:?}, array shape: {:?}, array strides: {:?}. array data len: {}",
                        index, self.iter.data_shape, self.data_strides, self.data_len
                    )
                }
                Some(index)
            }
        }
    }
}

fn increase(index: &mut [usize], shape: &[usize], counter: &mut usize, flag: &mut bool) {
    index[*counter] += 1;

    let mut axis_change = false;
    while index[*counter] >= shape[*counter] {
        axis_change = true;
        let len = index.len();
        index[*counter..len].iter_mut().for_each(|x| *x = 0);
        *counter = counter.wrapping_sub(1);
        if *counter < index.len() {
            index[*counter] += 1;
        } else {
            *flag = true;
            break;
        }
    }

    if axis_change {
        *counter = index.len() - 1;
    }
}

impl <'a> Iterator for NdArrayFastDataIndexIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_done {
            return None;
        }

        let ret = compute_index(&self.index, self.data_strides, self.data_offset);

        if ret >= self.data_len {
            panic!(
                "Index out of bounds. Index: {:?}, array shape: {:?}, array strides: {:?}. array data len: {}",
                self.index, self.data_shape, self.data_strides, self.data_len
            )
        }

        let ret = Some(ret);

        increase(&mut self.index, self.data_shape, &mut self.axis_counter, &mut self.has_done);

        ret
    }
}

impl <'a> Iterator for IndexIterator<'a> {
    type Item = NdArrayIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_done {
            return None;
        }

        let index = self.index.clone();

        increase(&mut self.index, self.data_shape, &mut self.axis_counter, &mut self.has_done);

        Some(index)
    }
}


/// # Example:
/// ```rust
/// use ndarray::array::NdArray;
///
/// // Example
/// let a = NdArray::new_shape(Vec::from_iter((1..28).into_iter()), vec![3, 3, 3]);
/// assert_eq!(a.into_iter().map(|x| *x).collect::<Vec<_>>(), Vec::from_iter((1..28).into_iter()));
/// ```
macro_rules! impl_nd_array_iter {
    ($( $type:ty ),+) => {
        $(
            impl <'a: 'b, 'b, T> IntoIterator for &'b $type {
                type Item = &'b T;
                type IntoIter = NdArrayIterator<'b, $type, T>;

                fn into_iter(self) -> Self::IntoIter {
                    NdArrayIterator::new(self)
                }
            }
        )+
    };
}

impl_nd_array_iter!{NdArray<T>, NdArrayView<'a, T>}