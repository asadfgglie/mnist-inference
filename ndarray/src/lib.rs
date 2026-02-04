use std::cmp::max;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use ndarray_marco::nd_array_index;

#[derive(Debug, PartialEq)]
pub struct NdArray<T> {
    data: Box<[T]>,
    shape: NdArrayIndex,
    strides: NdArrayIndex,
}

pub struct NdArrayView<'a, T> {
    data: &'a [T],
    shape: NdArrayIndex,
    strides: NdArrayIndex,
}

nd_array_index!{8}

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
    iter: IndexIterator<'a>
}

/// Use for NdArrayIterator prevents lots of Vec allocated.
///
/// This yields a valid physical index of linear memory data, not a logical index like `vec![1, 2, 0]`
pub struct NdArrayFastDataIndexIterator<'a> {
    data_len: usize,
    data_strides: &'a [usize],
    data_shape: &'a [usize],
    index: NdArrayIndex,
    axis_counter: usize,
    has_done: bool,
}

/// Iter logical index like `vec![1, 2, 0]` by shape.
///
/// Avoid using this for a big shape. This will cause lots of allocated.
struct IndexIterator<'a> {
    data_shape: &'a [usize],
    index: NdArrayIndex,
    axis_counter: usize,
    has_done: bool,
}

#[derive(Debug, PartialEq)]
pub enum NdArrayError {
    BroadcastError(String),
    InvalidStridesError(String),
    InvalidShapeError(String),
}

#[derive(Debug, PartialEq)]
pub struct Scalar<T>(pub T);

pub trait Cast<T> {
    type Target;
    fn cast(self) -> Self::Target;
}

pub trait NdArrayLike<T> {
    fn data<'a, 'b: 'a>(&'b self) -> &'a [T];
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn is_contiguous(&self) -> bool {
        self.strides() == compute_stride(self.shape())
    }
    fn compute_index(&self, indices: &[usize]) -> usize {
        let index = compute_index(indices, self.strides());
        match self.data().get(index) {
            Some(_) => (),
            None => {
                panic!("index {indices:?}=>({index}) out of bounds, shape: {:?}, stride: {:?}", self.shape(), self.strides())
            }
        };
        index
    }
    fn to_view<'a, 'b: 'a>(&'b self) -> NdArrayView<'a, T> {
        NdArrayView::new(self.data(), self.shape(), self.shape().into(), self.strides().into())
    }
    fn iter_index<'a>(&'a self) -> NdArrayDataIndexIterator<'a>
    where Self: Sized, T: 'a {
        NdArrayDataIndexIterator::new(self)
    }
    fn reshape<'a, 'b: 'a>(&'b self, shape: NdArrayIndex) -> NdArrayView<'a, T> {
        let stride = compute_broadcast_strides(self.shape(), self.strides(), &shape).into();
        NdArrayView::new(self.data(), self.shape(), shape, stride)
    }
    fn squeeze<'a, 'b: 'a>(&'b self, axis: usize) -> NdArrayView<'a, T> {
        match self.shape().get(axis) {
            Some(v) => {
                if *v != 1 {
                    self.to_view()
                } else {
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    self.reshape(shape.into())
                }
            },
            None => {
                panic!("Index out of bounds, shape: {:?}, axis: {axis}", self.shape())
            }
        }
    }
    fn unsqueeze<'a, 'b: 'a>(&'b self, axis: usize) -> NdArrayView<'a, T> {
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
                panic!("Index out of bounds, shape: {:?}, axis: {axis}", self.shape())
            }
        }
    }
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
}

impl <T> NdArray<T> {
    pub fn new_shape_with_strides(data: Box<[T]>, shape: NdArrayIndex, strides: NdArrayIndex) -> Self {
        if data.len() != shape.iter().product() {
            panic!("Data length ({}) does not match shape dimensions ({:?}), shape except {} length data.",
                   data.len(), shape, shape.iter().product::<usize>());
        }

        match validate_contiguous_stride(&shape, &strides) {
            Ok(_) => Self {
                data,
                shape: shape.into(),
                strides: strides.into(),
            },
            Err(_) => match validate_view(&shape, &shape, &strides) {
                Ok(_) => Self {
                    data,
                    shape: shape.into(),
                    strides: strides.into(),
                },
                Err(e) => panic!("{:?}", e)
            }
        }
    }

    pub fn new_shape_with_index(data: Vec<T>, shape: NdArrayIndex) -> Self {
        let stride = compute_stride(&shape);

        Self::new_shape_with_strides(
            data.into_boxed_slice(),
            shape,
            stride.into(),
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

    pub fn reshape_array(array: Self, shape: NdArrayIndex) -> Self {
        if array.data.len() != shape.iter().product() {
            panic!("Invalid shape ({:?}) does not match data len ({}).", shape, array.data.len());
        }
        let stride = compute_stride(&array.shape()).into();
        Self::new_shape_with_strides(array.data, shape, stride)
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

    pub fn item(self) -> Result<Scalar<T>, NdArrayError> {
        if self.shape != index![1] {
            return Err(NdArrayError::InvalidShapeError(format!(
                "shape must be 1 dimension 1 element array, but got {:?}", self.shape
            )))
        }
        let self_info = format!(
            "self.shape: {:?}, self.stride: {:?}, self.data.len: {}",
            self.shape, self.strides, self.data.len()
        );
        Ok(self.data.into_vec().pop().expect(&format!(
            "self.data seems empty, but this should not happen. {}", self_info
        )).into())
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
            self.strides = compute_stride(&self.shape).into();
        }
    }
}

impl <'a, T> NdArrayView<'a, T> {
    pub fn new<'b: 'a>(array: &'b [T], old_shape: &[usize], shape: NdArrayIndex, strides: NdArrayIndex) -> Self {
        match validate_view(old_shape, &shape, &strides) {
            Ok(_) => (),
            Err(e) => panic!("{:?}", e)
        }

        Self {
            data: array,
            shape,
            strides,
        }
    }
}

impl <'a, T: NdArrayLike<DT>, DT> NdArrayIterator<'a, T, DT> {
    pub fn new<'b: 'a>(array: &'b T) -> Self {
        Self {
            data: array,
            index_iter: NdArrayFastDataIndexIterator::iter_index(array.shape(), array.strides()),
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
        }
    }
}

impl <'a> NdArrayFastDataIndexIterator<'a> {
    pub fn iter_index<'b: 'a>(shape: &'b [usize], strides: &'b [usize]) -> Self {
        Self {
            data_len: shape.iter().product(),
            data_strides: strides,
            data_shape: shape,
            index: NdArrayIndex::zeros(shape.len()),
            axis_counter: shape.len() - 1,
            has_done: false,
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

fn compute_index(indices: &[usize], strides: &[usize]) -> usize {
    if indices.len() != strides.len() {
        panic!(
            "Indices length ({}) does not match NdArray dimensions ({})",
            indices.len(),
            strides.len()
        );
    }

    indices.iter()
        .enumerate()
        .fold(0, |acc, (axis, &x)| acc + x * strides[axis])
}

fn validate_view(old_shape: &[usize], view_shape: &[usize], view_stride: &[usize]) -> Result<(), NdArrayError> {
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
            if o == None && v == None && s == None {
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

fn validate_contiguous_stride(shape: &[usize], strides: &[usize]) -> Result<(), NdArrayError> {
    let contiguous_stride = compute_stride(shape);
    if strides == contiguous_stride {
        Ok(())
    } else {
        Err(NdArrayError::InvalidStridesError(
            format!("Except stride {:?}, got {:?}", contiguous_stride, strides)
        ))
    }
}

/// base on shape, return row-major contiguous stride
fn compute_stride(shape: &[usize]) -> Vec<usize> {
    shape.into_iter()
        .rev()
        .scan(1, |acc, x| {
            let tmp = *acc;
            *acc *= x;
            Some(tmp)
        })
        .collect::<Vec<usize>>()
        .into_iter()
        .rev()
        .collect()
}

/// ```rust
/// use ndarray::broadcast_shapes;
/// use ndarray::NdArrayError;
///
/// // Example 1: Broadcasting compatible shapes
/// let shape1 = vec![2, 3, 4];
/// let shape2 = vec![3, 1];
/// let result = broadcast_shapes(&shape1, &shape2).unwrap();
/// assert_eq!(result, vec![2, 3, 4]);
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
/// assert_eq!(result, vec![2, 3, 4, 5]);
///
/// // Example 4
/// let shape7 = vec![1];
/// let shape8 = vec![4];
/// let result = broadcast_shapes(&shape7, &shape8).unwrap();
/// assert_eq!(result, vec![4]);
/// ```
pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<NdArrayIndex, NdArrayError> {
    let longer = if lhs.len() > rhs.len() { lhs } else { rhs };
    let shorter = if lhs.len() > rhs.len() { rhs } else { lhs };

    let mut longer = longer.iter().rev();
    let mut shorter = shorter.iter().rev();

    let mut ret = Vec::with_capacity(longer.len());
    loop {
        let (l, r) = (longer.next(), shorter.next());

        if l == None && r == None {
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

fn compute_broadcast_strides(old_shape: &[usize], old_stride: &[usize], broadcast_shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = Vec::with_capacity(broadcast_shape.len());

    let mut old_shape_iter = old_shape.iter().rev();
    let mut broadcast_shape_iter = broadcast_shape.iter().rev();
    let mut old_stride_iter = old_stride.iter().rev();

    loop {
        let (o, b) = (old_shape_iter.next(), broadcast_shape_iter.next());

        if o == None && b == None {
            break;
        }

        let (o, b) = (match o {
            Some(x) => *x,
            None => 0 // hidden broadcast dimension, you can treat as unsqueeze axis
        }, match b {
            Some(x) => *x,
            None => panic!(
                "broadcast_shape shorter than old_array.shape, broadcast_shape len: {}, \
                                old_array.shape len: {}",
                broadcast_shape.len(),
                old_shape.len()
            )
        });

        if o == 1 || o == 0 && b > o { // this is the broadcast dimension,
            // so stride is always 0
            strides.push(0)
        } else {
            strides.push(*old_stride_iter.next().expect(
                "Unable to get old stride. old_stride_iter.next() is None."
            ))
        }
    }

    strides.reverse();
    strides
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
    match broadcast_shapes(&lhs.shape(), &rhs.shape()) {
        Ok(broadcast_shape) => {
            let rhs_shape = broadcast_shape.clone();
            let lhs_shape = broadcast_shape;

            let lhs_strides = compute_broadcast_strides(lhs.shape(), lhs.strides(), &lhs_shape);
            let rhs_strides = compute_broadcast_strides(rhs.shape(), rhs.strides(), &rhs_shape);

            Ok((NdArrayView::new(lhs.data(), lhs.shape(), lhs_shape.into(), lhs_strides.into()),
                NdArrayView::new(rhs.data(), rhs.shape(), rhs_shape.into(), rhs_strides.into())))
        }
        Err(e) => Err(e)
    }
}


// NdArray math op
// all NdArrayLike op will through broadcast_array() then become a `NdArrayView`
// so if &NdArrayView $op &NdArrayView is implement => NdArray, NdArrayView, NdArraySource
// and their & version also implement by those macros
// note that all op is element wise operation

/// basic `&NdArray`/`NdArray` operation with `&NdArray`/`NdArray`
macro_rules! op {
    ($( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+) => {
        $(
            impl <L, R> $op_trait<NdArray<R>> for NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: NdArray<R>) -> Self::Output {
                    &self $op &rhs
                }
            }
            impl <L, R> $op_trait<&NdArray<R>> for NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &NdArray<R>) -> Self::Output {
                    &self $op rhs
                }
            }
            impl <L, R> $op_trait<NdArray<R>> for &NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: NdArray<R>) -> Self::Output {
                    self $op &rhs
                }
            }
            impl <L, R> $op_trait<&NdArray<R>> for &NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &NdArray<R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    lhs $op rhs
                }
            }
        )+
    };
}

/// basic `&NdArrayLike`/`NdArrayLike` operation with `&NdArrayLike`/`NdArrayLike`
///
/// no define `&NdArrayView` operation with `&NdArrayView` (define by `ref_view_op!`)
macro_rules! ref_op {
    ( ( $( $type:ident ),+ ), $ops:tt ) => {
        $(
            ref_op!{$type, $ops}
        )+
    };
    ($type:ident, [ $( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+ ]) => {
        $(
            impl <'a, 'b, L, R> $op_trait<$type<'b, R>> for $type<'a, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: $type<'b, R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(&self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    &lhs $op &rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<&'c $type<'b, R>> for $type<'a, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'b: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &'c $type<'b, R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(&self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    &lhs $op rhs
                }
            }

            impl <'a, 'b, L, R> $op_trait<$type<'b, R>> for NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: $type<'b, R>) -> Self::Output {
                    &self $op &rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<&'c $type<'b, R>> for NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'b: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &'c $type<'b, R>) -> Self::Output {
                    &self $op rhs
                }
            }

            impl <'a, 'b, L, R> $op_trait<$type<'b, R>> for &NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: $type<'b, R>) -> Self::Output {
                    self $op &rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<&'c $type<'b, R>> for &NdArray<L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'b: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &'c $type<'b, R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(self, rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    &lhs $op &rhs
                }
            }

            impl <'a, 'b, L, R> $op_trait<NdArray<R>> for $type<'b, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: NdArray<R>) -> Self::Output {
                    &self $op &rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<NdArray<R>> for &'c $type<'b, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'b: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: NdArray<R>) -> Self::Output {
                    self $op &rhs
                }
            }

            impl <'a, 'b, L, R> $op_trait<&NdArray<R>> for $type<'b, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &NdArray<R>) -> Self::Output {
                    &self $op rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<&NdArray<R>> for &'c $type<'b, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'b: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &NdArray<R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(self, rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    lhs $op rhs
                }
            }

            impl <'a, 'b, 'c, L, R> $op_trait<$type<'b, R>> for &'c $type<'a, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'a: 'c {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: $type<'b, R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(&self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    lhs $op &rhs
                }
            }
        )+
    };
}

/// define `&NdArrayView` operation with `&NdArrayView` as core op implementation
///
/// all op based on this implementation
macro_rules! ref_view_op {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        $(
            impl <'a, 'b, 'c, 'd, L, R> $op_trait<&'d NdArrayView<'b, R>> for &'c NdArrayView<'a, L>
            where L: $op_trait<Output=L> + Clone, R: Into<L> + Clone, 'a: 'c, 'b: 'd {
                type Output = NdArray<L>;

                fn $op_fn(self, rhs: &'d NdArrayView<'b, R>) -> Self::Output {
                    let (lhs, rhs) = match broadcast_array(self, rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    let mut data: Vec<L> = Vec::with_capacity(lhs.shape.iter().product());

                    for (l, r) in zip(lhs.into_iter(), rhs.into_iter()) {
                        let (l, r) = (l.clone(), r.clone());
                        data.push(l $op r.into())
                    }

                    Self::Output::new_shape_with_index(data, lhs.shape)
                }
            }
        )+
    };
}

/// combine `op!`, `ref_view_op!`, `ref_op!`
///
/// auto implement `NdArrayView` marco
macro_rules! general_op {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        op!{$( ($op, $op_trait, $op_fn) ),+}
        ref_view_op!{$( ($op, $op_trait, $op_fn) ),+}
        ref_op!{(NdArrayView), [$( ($op, $op_trait, $op_fn) ),+]}
    };
}

/// basic `&NdArray`/`NdArray` assign (in-place) operation with `NdArray`
/// ```
/// use ndarray::*;
/// fn main() {
///     let mut a: NdArray<f32> = NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast();
///     a += NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]);
///     assert_eq!(a, NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast() * Scalar(2i8));
///
///     a /= Scalar(2i8);
///     assert_eq!(a, NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast());
///
///     a -= NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]);
///     assert_eq!(a, NdArray::new_shape(vec![0.0; 9], vec![3,3]));
/// }
/// ```
macro_rules! assign_op {
    ($( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+) => {
        $(
            impl <L, R> $op_trait<NdArray<R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: NdArray<R>) {
                    <NdArray<L> as $op_trait<&NdArray<R>>>::$op_fn(self, &rhs);
                }
            }
            impl <L, R> $op_trait<&NdArray<R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: &NdArray<R>) {
                    let (lhs, rhs) = match broadcast_array(self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    if lhs.shape() != self.shape() {
                        panic!(
                            "self shape {:?} can't broadcast when operate += operator, which is cause by rhs shape {:?}",
                            self.shape(), rhs.shape()
                        )
                    }

                    self.contiguous_self();

                    let iter: Vec<NdArrayIndex> = self.iter_index().collect();
                    for indices in iter {
                        let (self_index, rhs_index) = (self.compute_index(&indices), rhs.compute_index(&indices));
                        self.data[self_index] $op rhs.data()[rhs_index].clone().into();
                    }
                }
            }
        )+
    };
}

/// basic `&NdArrayLike` assign (in-place) operation with `NdArray`
macro_rules! ref_assign_op {
    (( $( $type:ident ),+ ), $ops:tt) => {
        $(
            ref_assign_op!{$type, $ops}
        )+
    };
    ($type:ident, [ $( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+ ]) => {
        $(
            impl <L, R> $op_trait<$type<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: $type<'_, R>) {
                    <NdArray<L> as $op_trait<&$type<'_, R>>>::$op_fn(self, &rhs);
                }
            }
            impl <L, R> $op_trait<&$type<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: &$type<'_, R>) {
                    let (lhs, rhs) = match broadcast_array(self, &rhs) {
                        Ok((lhs, rhs)) => (lhs, rhs),
                        Err(e) => panic!("{:?}", e)
                    };

                    if lhs.shape() != self.shape() {
                        panic!(
                            "self shape {:?} can't broadcast when operate += operator, which is cause by rhs shape {:?}",
                            self.shape(), rhs.shape()
                        )
                    }

                    self.contiguous_self();

                    let iter: Vec<NdArrayIndex> = self.iter_index().collect();
                    for indices in iter {
                        let (self_index, rhs_index) = (self.compute_index(&indices), rhs.compute_index(&indices));
                        self.data[self_index] $op rhs.data()[rhs_index].clone().into();
                    }
                }
            }
        )+
    };
}

/// combine `assign_op!`, `ref_assign_op!`
///
/// auto implement `NdArrayView` marco
macro_rules! general_assign_op {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        assign_op!{$( ($op, $op_trait, $op_fn) ),+}
        ref_assign_op!{(NdArrayView), [ $( ($op, $op_trait, $op_fn) ),+ ] }
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
/// 
/// but only `*` satisfy commutative, which can ExChange Order (eco)
/// 
/// here we only implement a meaningful order: `&NdArray`/`NdArray` `op` `Scalar`
macro_rules! no_eco_op_scalar {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        $(
            eco_op_scalar!{$op, $op_trait, $op_fn}
        )+
    };
    ($op:tt, $op_trait:ident, $op_fn:ident) => {
        impl <T, ST> $op_trait<Scalar<ST>> for NdArray<T>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: Scalar<ST>) -> Self::Output {
                &self $op &rhs
            }
        }

        impl <T, ST> $op_trait<&Scalar<ST>> for NdArray<T>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &Scalar<ST>) -> Self::Output {
                &self $op rhs
            }
        }

        impl <T, ST> $op_trait<Scalar<ST>> for &NdArray<T>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: Scalar<ST>) -> Self::Output {
                self $op &rhs
            }
        }

        impl <T, ST> $op_trait<&Scalar<ST>> for &NdArray<T>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &Scalar<ST>) -> Self::Output {
                self $op <Scalar<ST> as Into<NdArray<ST>>>::into(Scalar(rhs.0.clone()))
            }
        }
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
///
/// but only `*` satisfy commutative, which can ExChange Order (eco)
///
/// here we implement the other meaningful order: `Scalar` `op` `&NdArray`/`NdArray`
macro_rules! eco_op_scalar {
    ($op:tt, $op_trait:ident, $op_fn:ident) => {
        no_eco_op_scalar!{$op, $op_trait, $op_fn}

        impl <T, ST> $op_trait<NdArray<T>> for Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: NdArray<T>) -> Self::Output {
                &rhs $op &self
            }
        }

        impl <T, ST> $op_trait<NdArray<T>> for &Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: NdArray<T>) -> Self::Output {
                &rhs $op self
            }
        }

        impl <T, ST> $op_trait<&NdArray<T>> for Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &NdArray<T>) -> Self::Output {
                rhs $op &self
            }
        }

        impl <T, ST> $op_trait<&NdArray<T>> for &Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &NdArray<T>) -> Self::Output {
                rhs $op <Scalar<ST> as Into<NdArray<ST>>>::into(Scalar(self.0.clone()))
            }
        }
    };
    ([ $( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+ ]) => {
        $(
            eco_op_scalar!{$op, $op_trait, $op_fn}
        )+
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
///
/// but only `*` satisfy commutative, which can ExChange Order (eco)
///
/// here we only implement a meaningful order: `&NdArrayLike`/`NdArrayLike` `op` `Scalar`
macro_rules! ref_no_eco_op_scalar {
    (( $( $type:ident ),+ ), $ops:tt) => {
        $(
            ref_no_eco_op_scalar!{$type, $ops}
        )+
    };
    ($type:ident, [ $( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+ ]) => {
        $(
            impl <'a, T, ST> $op_trait<Scalar<ST>> for $type<'a, T>
            where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
                type Output = NdArray<T>;

                fn $op_fn(self, rhs: Scalar<ST>) -> Self::Output {
                    &self $op &rhs
                }
            }

            impl <'a, T, ST> $op_trait<&Scalar<ST>> for $type<'a, T>
            where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
                type Output = NdArray<T>;

                fn $op_fn(self, rhs: &Scalar<ST>) -> Self::Output {
                    &self $op rhs
                }
            }

            impl <'a, T, ST> $op_trait<Scalar<ST>> for &$type<'a, T>
            where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
                type Output = NdArray<T>;

                fn $op_fn(self, rhs: Scalar<ST>) -> Self::Output {
                    self $op &rhs
                }
            }

            impl <'a, T, ST> $op_trait<&Scalar<ST>> for &$type<'a, T>
            where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
                type Output = NdArray<T>;

                fn $op_fn(self, rhs: &Scalar<ST>) -> Self::Output {
                    self $op <Scalar<ST> as Into<NdArray<ST>>>::into(Scalar(rhs.0.clone()))
                }
            }
        )+
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
///
/// but only `*` satisfy commutative, which can ExChange Order (eco)
///
/// here we implement the other meaningful order: `Scalar` `op` `&NdArrayLike`/`NdArrayLike`
macro_rules! ref_eco_op_scalar {
    (( $( $type:ident ),+ ), $ops:tt) => {
        $(
            ref_eco_op_scalar!{$type, $ops}
        )+
    };
    ($type:ident, [($op:tt, $op_trait:ident, $op_fn:ident)]) => {
        ref_no_eco_op_scalar!{$type, [($op, $op_trait, $op_fn)]}

        impl <'a, T, ST> $op_trait<$type<'a, T>> for Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: $type<'a, T>) -> Self::Output {
                &rhs $op &self
            }
        }

        impl <'a, T, ST> $op_trait<$type<'a, T>> for &Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: $type<'a, T>) -> Self::Output {
                &rhs $op self
            }
        }

        impl <'a, T, ST> $op_trait<&$type<'a, T>> for Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &$type<'a, T>) -> Self::Output {
                rhs $op &self
            }
        }

        impl <'a, T, ST> $op_trait<&$type<'a, T>> for &Scalar<ST>
        where T: $op_trait<Output=T> + Clone, ST: Into<T> + Clone {
            type Output = NdArray<T>;

            fn $op_fn(self, rhs: &$type<'a, T>) -> Self::Output {
                rhs $op <Scalar<ST> as Into<NdArray<ST>>>::into(Scalar(self.0.clone()))
            }
        }
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
///
/// but only `*` satisfy commutative, which can ExChange Order (eco)
///
/// here we only implement a meaningful order: `&NdArrayLike`/`NdArrayLike` `op` `Scalar`
///
/// auto implement `NdArrayView` marco
macro_rules! general_no_eco_op_scalar {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        no_eco_op_scalar!{$( ($op, $op_trait, $op_fn) ),+}
        ref_no_eco_op_scalar!{(NdArrayView), [$( ($op, $op_trait, $op_fn) ),+]}
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
///
/// but only `*` satisfy commutative, which can ExChange Order (eco)
///
/// here we implement the other meaningful order: `Scalar` `op` `&NdArrayLike`/`NdArrayLike`
///
/// auto implement `NdArrayView` marco
macro_rules! general_eco_op_scalar {
    ($op:tt, $op_trait:ident, $op_fn:ident) => {
        eco_op_scalar!{$op, $op_trait, $op_fn}
        ref_eco_op_scalar!{(NdArrayView), [($op, $op_trait, $op_fn)]}
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
macro_rules! assign_scalar_op {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        $(
            impl <T, ST> $op_trait<Scalar<ST>> for NdArray<T>
            where T: $op_trait + Clone, ST: Into<T> + Clone {
                fn $op_fn(&mut self, rhs: Scalar<ST>) {
                    <NdArray<T> as $op_trait<&Scalar<ST>>>::$op_fn(self, &rhs)
                }
            }

            impl <T, ST> $op_trait<&Scalar<ST>> for NdArray<T>
            where T: $op_trait + Clone, ST: Into<T> + Clone {
                fn $op_fn(&mut self, rhs: &Scalar<ST>) {
                    self.contiguous_self();
                    self.data.iter_mut().for_each(|x| {
                        let rhs = rhs.0.clone();
                        *x $op rhs.into();
                    });
                }
            }
        )+
    };

}

general_op!{(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
general_assign_op!{
    (+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign), 
    (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)
}
general_no_eco_op_scalar!{(%, Rem, rem), (/, Div, div)}
general_eco_op_scalar!{*, Mul, mul}
assign_scalar_op!{(*=, MulAssign, mul_assign), (%=, RemAssign, rem_assign), (/=, DivAssign, div_assign)}

pub fn matmul<L: Clone + Mul<Output=L>, R: Into<L> + Clone>(lhs: & impl NdArrayLike<L>, rhs: & impl NdArrayLike<R>) -> NdArray<L> {
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
                    ret.push(v * tmp);
                }
            }
        }

        assert_eq!(ret.len(), 1);
        NdArray::new(ret)
    } else if lhs.shape().len() == 1 && rhs.shape().len() > 1 {
        let ret = matmul(&lhs.reshape(index![1, lhs.shape()[0]]), rhs);
        let axis = ret.shape().len() - 2;
        NdArray::squeeze_array(ret, axis)
    } else if lhs.shape().len() > 1 && rhs.shape().len() == 1 {
        let ret = matmul(lhs, &rhs.reshape(index![rhs.shape()[0], 1]));
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

        let lhs = lhs.reshape(broadcast_batch_shape.clone().concat(lhs_data_shape.into()));
        let rhs = rhs.reshape(broadcast_batch_shape.clone().concat(rhs_data_shape.into()));
        // lhs: (broadcast_batch_shape, M, P)
        // rhs: (broadcast_batch_shape, P, N)

        let mut data_shape = broadcast_batch_shape.to_vec();
        data_shape.push(m);
        data_shape.push(n);
        let data_shape: NdArrayIndex = data_shape.into();
        let mut data: Vec<L> = Vec::with_capacity(data_shape.iter().product());
        let data_stride: NdArrayIndex = compute_stride(&data_shape).into();

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

                        let tmp = lhs.data()[lhs.compute_index(&lhs_index)].clone() * rhs.data()[rhs.compute_index(&rhs_index)].clone().into();
                        match tmp_ret.is_empty() {
                            true => tmp_ret.push(tmp),
                            false => {
                                let v = tmp_ret.pop().unwrap();
                                tmp_ret.push(v * tmp);
                            }
                        }
                    }

                    assert_eq!(tmp_ret.len(), 1);
                    data[compute_index(&data_index, &data_stride)] = tmp_ret.pop().unwrap();
                }
            }
        }

        NdArray::new_shape_with_strides(data.into_boxed_slice(), data_shape, data_stride)
    } else {
        panic!(
            "Unexcept shape, lhs shape len and rhs shape len must greater than 0. lhs shape: {:?}, rhs shape: {:?}",
            lhs.shape(), rhs.shape()
        )
    }
}


// Scalar math op
impl <L, R> Add<Scalar<R>> for Scalar<L>
where L: Add<Output=L>, R: Into<L> {
    type Output = Self;

    fn add(self, rhs: Scalar<R>) -> Self::Output {
        Self(self.0 + rhs.0.into())
    }
}

impl <L, R> Sub<Scalar<R>> for Scalar<L>
where L: Sub<Output=L>, R: Into<L> {
    type Output = Self;

    fn sub(self, rhs: Scalar<R>) -> Self {
        Self(self.0 - rhs.0.into())
    }
}

impl <L, R> Mul<Scalar<R>> for Scalar<L>
where L: Mul<Output=L>, R: Into<L> {
    type Output = Self;

    fn mul(self, rhs: Scalar<R>) -> Self::Output {
        Self(self.0 * rhs.0.into())
    }
}

impl <L, R> Div<Scalar<R>> for Scalar<L>
where L: Div<Output=L>, R: Into<L> {
    type Output = Self;

    fn div(self, rhs: Scalar<R>) -> Self::Output {
        Self(self.0 / rhs.0.into())
    }
}

impl <L, R> Rem<Scalar<R>> for Scalar<L>
where L: Rem<Output=L>, R: Into<L> {
    type Output = Self;

    fn rem(self, rhs: Scalar<R>) -> Self::Output {
        Self(self.0 % rhs.0.into())
    }
}

impl <L, R> AddAssign<Scalar<R>> for Scalar<L>
where L: AddAssign, R: Into<L> {
    fn add_assign(&mut self, rhs: Scalar<R>) {
        self.0 += rhs.0.into()
    }
}

impl <L, R> SubAssign<Scalar<R>> for Scalar<L>
where L: SubAssign, R: Into<L> {
    fn sub_assign(&mut self, rhs: Scalar<R>) {
        self.0 -= rhs.0.into()
    }
}

impl <L, R> MulAssign<Scalar<R>> for Scalar<L>
where L: MulAssign, R: Into<L> {
    fn mul_assign(&mut self, rhs: Scalar<R>) {
        self.0 *= rhs.0.into()
    }
}

impl <L, R> DivAssign<Scalar<R>> for Scalar<L>
where L: DivAssign, R: Into<L> {
    fn div_assign(&mut self, rhs: Scalar<R>) {
        self.0 /= rhs.0.into()
    }
}

impl <L, R> RemAssign<Scalar<R>> for Scalar<L>
where L: RemAssign, R: Into<L> {
    fn rem_assign(&mut self, rhs: Scalar<R>) {
        self.0 %= rhs.0.into()
    }
}


// Indexing
impl <T, Idx> Index<Idx> for NdArray<T>
where Idx: Into<NdArrayIndex>{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        self.data.index(self.compute_index(&index.into()))
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
                let i = compute_index(&index, self.data_strides);
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

        let ret = compute_index(&self.index, self.data_strides);

        if ret >= self.data_len {
            panic!(
                "Index out of bounds. Index: {:?}, array shape: {:?}, array strides: {:?}. array data len: {}",
                self.index, self.data_shape, self.data_strides, self.data_len
            )
        }

        let ret = Some(ret);

        increase(&mut self.index, &self.data_shape, &mut self.axis_counter, &mut self.has_done);

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

        increase(&mut self.index, &self.data_shape, &mut self.axis_counter, &mut self.has_done);

        Some(index)
    }
}


/// # Example:
/// ```rust
/// use ndarray::NdArray;
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


// T conversions
impl <T: Clone> From<Vec<T>> for NdArray<T> {
    fn from(data: Vec<T>) -> Self {
        NdArray::new(data)
    }
}

impl <T> From<Scalar<T>> for NdArray<T> {
    fn from(data: Scalar<T>) -> Self {
        NdArray::new_shape_with_strides(Box::new([data.0]), index![1], index![1])
    }
}

impl <T> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl <T> Deref for Scalar<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl <T> DerefMut for Scalar<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl <T, NT: Clone> Cast<NT> for NdArray<T>
where NT: From<T> {
    type Target = NdArray<NT>;

    fn cast(self) -> Self::Target {
        NdArray::new_shape_with_strides(
            self.data.into_iter().map(|x| x.into()).collect(),
            self.shape,
            self.strides,
        )
    }
}

impl <T, NT> Cast<NT> for Scalar<T>
where T: Into<NT> {
    type Target = Scalar<NT>;

    fn cast(self) -> Self::Target {
        Scalar(self.0.into())
    }
}

impl <T: Clone> From<NdArrayView<'_, T>> for NdArray<T> {
    fn from(value: NdArrayView<T>) -> Self {
        let mut data = Vec::with_capacity(value.shape.iter().product());
        data.extend(value.into_iter().cloned());
        Self::new_shape_with_index(data, value.shape)
    }
}