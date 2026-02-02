use std::cmp::max;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign, Deref, DerefMut};

#[derive(Debug, PartialEq)]
pub struct NdArray<T> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    is_contiguous: bool
}

#[derive(Clone, Copy)]
pub enum NdArraySource<'a, T> {
    Owned(&'a NdArray<T>),
    View(&'a dyn NdArrayLike<T>)
}

pub struct NdArrayView<'a, T> {
    data: NdArraySource<'a, T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

pub struct NdArrayIterator<'a, T> {
    data: &'a dyn NdArrayLike<T>,
    index_iter: NdArrayDataIndexIterator<'a>,
}

// Iter index by NdArray shape and stride
pub struct NdArrayDataIndexIterator<'a> {
    ref_data_len: usize,
    ref_data_shape: &'a [usize],
    ref_data_strides: &'a [usize],
    index: Vec<usize>,
    axis_counter: usize,
    has_done: bool,
}

#[derive(Debug, PartialEq)]
pub enum NdArrayError {
    BroadcastError(String),
    InvalidStridesError(String),
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
    fn is_contiguous(&self) -> bool;
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
    fn to_view<'a, 'b: 'a>(&'b self) -> NdArrayView<'a, T>;
    fn iter_index<'a>(&'a self) -> NdArrayDataIndexIterator<'a>
    where Self: Sized, T: 'a {
        NdArrayDataIndexIterator::new(self)
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
    fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }
    fn to_view<'a, 'b: 'a>(&'b self) -> NdArrayView<'a, T> {
        NdArrayView::new(NdArraySource::Owned(self), self.shape.clone(), self.strides.clone())
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
    fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }
    fn to_view<'a, 'b: 'a>(&'b self) -> NdArrayView<'a, T> {
        NdArrayView::new(NdArraySource::Owned(self), self.shape.clone(), self.strides.clone())
    }
}

impl <'a, T> NdArrayLike<T> for NdArrayView<'a, T> {
    fn data<'b, 'c: 'b>(&'c self) -> &'b [T] {
        self.data.data()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn is_contiguous(&self) -> bool {
        self.shape == self.data.shape()
    }
    fn to_view<'b, 'c: 'b>(&'c self) -> NdArrayView<'b, T> {
        NdArrayView::new(NdArraySource::View(self), self.shape.clone(), self.strides.clone())
    }
}

impl <'a: 'b, 'b, T> NdArrayLike<T> for &'b NdArrayView<'a, T> {
    fn data<'c ,'d: 'c>(&'d self) -> &'c [T] {
        self.data.data()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    fn is_contiguous(&self) -> bool {
        self.shape == self.data.shape()
    }
    fn to_view<'c ,'d: 'c>(&'d self) -> NdArrayView<'c, T> {
        NdArrayView::new(NdArraySource::View(self), self.shape.clone(), self.strides.clone())
    }
}

impl <'a, T> NdArrayLike<T> for NdArraySource<'a, T> {
    fn data<'b, 'c: 'b>(&'c self) -> &'b [T] {
        match self {
            NdArraySource::Owned(array) => &array.data,
            NdArraySource::View(view) => view.data()
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            NdArraySource::Owned(array) => &array.shape,
            NdArraySource::View(view) => view.shape()
        }
    }
    fn strides(&self) -> &[usize] {
        match self {
            NdArraySource::Owned(array) => &array.strides,
            NdArraySource::View(view) => view.strides()
        }
    }
    fn is_contiguous(&self) -> bool {
        match self {
            NdArraySource::Owned(array) => array.is_contiguous,
            NdArraySource::View(view) => view.is_contiguous()
        }
    }
    fn to_view<'b, 'c: 'b>(&'c self) -> NdArrayView<'b, T> {
        match self {
            NdArraySource::Owned(array) => array.to_view(),
            NdArraySource::View(view) => view.to_view()
        }
    }
}

impl <'a: 'b, 'b, T> NdArrayLike<T> for &'b NdArraySource<'a, T> {
    fn data<'c, 'd: 'c>(&'d self) -> &'c [T] {
        match self {
            NdArraySource::Owned(array) => &array.data,
            NdArraySource::View(view) => view.data()
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            NdArraySource::Owned(array) => &array.shape,
            NdArraySource::View(view) => view.shape()
        }
    }
    fn strides(&self) -> &[usize] {
        match self {
            NdArraySource::Owned(array) => &array.strides,
            NdArraySource::View(view) => view.strides()
        }
    }
    fn is_contiguous(&self) -> bool {
        match self {
            NdArraySource::Owned(array) => array.is_contiguous,
            NdArraySource::View(view) => view.is_contiguous()
        }
    }
    fn to_view<'c, 'd: 'c>(&'d self) -> NdArrayView<'c, T> {
        match self {
            NdArraySource::Owned(array) => array.to_view(),
            NdArraySource::View(view) => view.to_view()
        }
    }
}

impl <T> NdArray<T> {
    pub fn new_shape_with_strides(data: Box<[T]>, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        if data.len() != shape.iter().product() {
            panic!("Data length ({}) does not match shape dimensions ({:?}), shape except {} length data.",
                   data.len(), shape, shape.iter().product::<usize>());
        }

        match validate_contiguous_stride(&shape, &strides) {
            Ok(_) => Self {
                data,
                shape,
                strides,
                is_contiguous: true
            },
            Err(_) => match validate_view(&shape, &shape, &strides) {
                Ok(_) => Self {
                    data,
                    shape,
                    strides,
                    is_contiguous: false
                },
                Err(e) => panic!("{:?}", e)
            }
        }
    }

    pub fn new_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        let stride = compute_stride(&shape);

        Self::new_shape_with_strides(
            data.into_boxed_slice(),
            shape,
            stride,
        )
    }

    pub fn new(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::new_shape(data, shape)
    }

    pub fn item(self) -> Scalar<T> {
        if self.shape != vec![1] {
            panic!(
                "shape must be 1 dimension 1 element array, but got {:?}", self.shape
            )
        }
        let self_info = format!(
            "self.shape: {:?}, self.stride: {:?}, self.data.len: {}",
            self.shape, self.strides, self.data.len()
        );
        self.data.into_vec().pop().expect(&format!(
            "self.data seems empty, but this should not happen. {}", self_info
        )).into()
    }
}

impl <T: Clone> NdArray<T> {
    pub fn contiguous(self) -> Self {
        if self.is_contiguous {
            self
        } else {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());

            self.into_iter()
                .for_each(|x| data.push(x.clone()));
            Self::new_shape(data, self.shape.clone())
        }
    }

    pub fn contiguous_self(& mut self) {
        if !self.is_contiguous {
            let mut data: Vec<T> = Vec::with_capacity(self.shape.iter().product());

            self.into_iter()
                .for_each(|x| data.push(x.clone()));
            self.data = data.into_boxed_slice();
            self.is_contiguous = true;
        }
    }
}

impl <'a, T> NdArrayView<'a, T> {
    pub fn new(array: NdArraySource<'a, T>, like: Vec<usize>, strides: Vec<usize>) -> Self {
        match validate_view(&array.shape(), &like, &strides) {
            Ok(_) => (),
            Err(e) => panic!("{:?}", e)
        }

        Self {
            data: array,
            shape: like,
            strides,
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

impl <'a, T> NdArrayIterator<'a, T> {
    pub fn new<'b: 'a>(array: &'b dyn NdArrayLike<T>) -> Self {
        Self {
            data: array,
            index_iter: NdArrayDataIndexIterator::new(array),
        }
    }
}

impl <'a> NdArrayDataIndexIterator<'a> {
    pub fn new<'b: 'a, T>(array: &'b dyn NdArrayLike<T>) -> Self {
        let rank = array.shape().len();

        Self {
            ref_data_len: array.data().len(),
            ref_data_shape: array.shape(),
            ref_data_strides: array.strides(),
            index: vec![0; rank],
            axis_counter: rank - 1,
            has_done: false,
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
                    "Strides {:?} are invalid for a view of shape {:?}, old shape: {:?}",
                    view_stride, view_shape, old_shape
                )))
            }
        }
        Ok(())
    }
}

fn validate_contiguous_stride(shape: &[usize], strides: &[usize]) -> Result<(), NdArrayError> {
    if shape.len() != strides.len() {
        Err(NdArrayError::InvalidStridesError(format!(
            "Shape length ({}) does not match strides length ({})", shape.len(), strides.len()
        )))
    } else {
        for (index, stride) in strides.iter().enumerate() {
            if index == shape.len() - 1 {
                if *stride != 1 {
                    return Err(NdArrayError::InvalidStridesError(format!(
                        "strides[{index}] except 1, but got {stride}, strides: {strides:?}, shape: {shape:?}"
                    )))
                }
            }
            let except_stride = shape[(index + 1)..shape.len()].iter().product::<usize>();
            if *stride != except_stride {
                return Err(NdArrayError::InvalidStridesError(format!(
                    "strides[{index}] except {except_stride}, but got {stride}, strides: {strides:?}, shape: {shape:?}"
                )))
            }
        }
        Ok(())
    }
}

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
/// use mnist_inference::broadcast_shapes;
/// use mnist_inference::NdArrayError;
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
pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, NdArrayError> {
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
    Ok(ret)
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
/// use mnist_inference::{broadcast_array, NdArray};
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
pub fn broadcast_array<'a, 'b, L, R, LT, RT>(lhs: &'a LT, rhs: &'b RT)
    -> Result<(NdArrayView<'a, L>, NdArrayView<'b, R>), NdArrayError>
where LT: NdArrayLike<L>, RT: NdArrayLike<R>, {
    match broadcast_shapes(&lhs.shape(), &rhs.shape()) {
        Ok(batch_shape) => {
            let rhs_shape = batch_shape.clone();
            let lhs_shape = batch_shape;

            fn compute_strides<T, DT>(old_array: &T, broadcast_shape: &[usize]) -> Vec<usize>
            where T: NdArrayLike<DT>, {
                let mut strides: Vec<usize> = Vec::with_capacity(broadcast_shape.len());

                let mut old_shape_iter = old_array.shape().iter().rev();
                let mut broadcast_shape_iter = broadcast_shape.iter().rev();
                let mut old_stride_iter = old_array.strides().iter().rev();

                loop {
                    let (o, b) = (old_shape_iter.next(), broadcast_shape_iter.next());

                    if o == None && b == None {
                        break;
                    }

                    let (o, b) = (match o {
                        Some(x) => *x,
                        None => 0 // hidden broadcast dimension
                    }, match b {
                        Some(x) => *x,
                        None => panic!(
                            "broadcast_shape shorter than old_array.shape, broadcast_shape len: {}, \
                                old_array.shape len: {}",
                            broadcast_shape.len(),
                            old_array.shape().len()
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

            let rhs_strides = compute_strides(lhs, &rhs_shape);
            let lhs_strides = compute_strides(rhs, &lhs_shape);

            Ok((NdArrayView::new(NdArraySource::View(lhs), rhs_shape, rhs_strides),
                NdArrayView::new(NdArraySource::View(rhs), lhs_shape, lhs_strides)))
        },
        Err(e) => Err(e)
    }
}


// NdArray math op
// all NdArrayLike op will through broadcast_array() then become a `NdArrayView`
// so if &NdArrayView $op &NdArrayView is implement => NdArray, NdArrayView, NdArraySource
// and their & version also implement by those macros
// note that all op is element wise operation
macro_rules! nd_array_op {
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

macro_rules! nd_array_assign_op {
    ($( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+) => {
        $(
            /// ```
            /// use mnist_inference::*;
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

                    let iter: Vec<Vec<usize>> = self.iter_index().collect();
                    for indices in iter {
                        let (self_index, rhs_index) = (self.compute_index(&indices), rhs.compute_index(&indices));
                        self.data[self_index] $op rhs.data()[rhs_index].clone().into();
                    }
                }
            }
            impl <L, R> $op_trait<NdArrayView<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: NdArrayView<'_, R>) {
                    <NdArray<L> as $op_trait<&NdArrayView<'_, R>>>::$op_fn(self, &rhs);
                }
            }
            impl <L, R> $op_trait<&NdArrayView<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: &NdArrayView<'_, R>) {
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

                    let iter: Vec<Vec<usize>> = self.iter_index().collect();
                    for indices in iter {
                        let (self_index, rhs_index) = (self.compute_index(&indices), rhs.compute_index(&indices));
                        self.data[self_index] $op rhs.data()[rhs_index].clone().into();
                    }
                }
            }
            impl <L, R> $op_trait<NdArraySource<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: NdArraySource<'_, R>) {
                    <NdArray<L> as $op_trait<&NdArraySource<'_, R>>>::$op_fn(self, &rhs);
                }
            }
            impl <L, R> $op_trait<&NdArraySource<'_, R>> for NdArray<L>
            where L: $op_trait + Clone, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: &NdArraySource<'_, R>) {
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

                    let iter: Vec<Vec<usize>> = self.iter_index().collect();
                    for indices in iter {
                        let (self_index, rhs_index) = (self.compute_index(&indices), rhs.compute_index(&indices));
                        self.data[self_index] $op rhs.data()[rhs_index].clone().into();
                    }
                }
            }
        )+
    };
}

macro_rules! nd_array_ref_op {
    ( ( $( $type:ident ),+ ), $ops:tt ) => {
        $(
            nd_array_ref_op!{$type, $ops}
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

macro_rules! nd_array_view_ref_op {
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

                    let shape = lhs.shape.to_vec();
                    let stride = lhs.strides.to_vec();

                    let mut data: Vec<L> = Vec::with_capacity(lhs.shape.iter().product());

                    for (l, r) in zip(lhs.into_iter(), rhs.into_iter()) {
                        let (l, r) = (l.clone(), r.clone());
                        data.push(l $op r.into())
                    }

                    Self::Output::new_shape_with_strides(data.into_boxed_slice(), shape, stride)
                }
            }
        )+
    };
}

macro_rules! nd_array_general_op {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        nd_array_op!{$( ($op, $op_trait, $op_fn) ),+}
        nd_array_view_ref_op!{$( ($op, $op_trait, $op_fn) ),+}
        nd_array_ref_op!{(NdArrayView, NdArraySource), [$( ($op, $op_trait, $op_fn) ),+]}
    };
}

macro_rules! nd_array_with_non_commutative_op_scalar {
    ($( ($op:tt, $op_trait:ident, $op_fn:ident) ),+) => {
        $(
            nd_array_with_non_commutative_op_scalar!{$op, $op_trait, $op_fn}
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

macro_rules! nd_array_with_commutative_op_scalar {
    ($op:tt, $op_trait:ident, $op_fn:ident) => {
        nd_array_with_non_commutative_op_scalar!{$op, $op_trait, $op_fn}

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
}

macro_rules! nd_array_ref_with_non_commutative_op_scalar {
    ($type:ident, $op:tt, $op_trait:ident, $op_fn:ident) => {
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
    };
}

macro_rules! nd_array_ref_with_commutative_op_scalar {
    ($type:ident, $op:tt, $op_trait:ident, $op_fn:ident) => {
        nd_array_ref_with_non_commutative_op_scalar!{$type, $op, $op_trait, $op_fn}

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

nd_array_general_op!{(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
nd_array_assign_op!{(+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign), (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)}
nd_array_with_non_commutative_op_scalar!{(%, Rem, rem), (/, Div, div)}
nd_array_with_commutative_op_scalar!{*, Mul, mul}
nd_array_ref_with_non_commutative_op_scalar!{NdArrayView, *, Mul, mul}

impl <T, ST> MulAssign<Scalar<ST>> for NdArray<T>
where T: MulAssign, ST: Into<T> + Clone {
    fn mul_assign(&mut self, rhs: Scalar<ST>) {
        self.data.iter_mut()
            .for_each(|x| *x *= rhs.0.clone().into());
    }
}

impl <T, ST> DivAssign<Scalar<ST>> for NdArray<T>
where T: DivAssign, ST: Into<T> + Clone {
    fn div_assign(&mut self, rhs: Scalar<ST>) {
        self.data.iter_mut()
            .for_each(|x| *x /= rhs.0.clone().into());
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
where T: Add<Output=T>, Idx: Into<Vec<usize>>{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        self.data.index(self.compute_index(&index.into()))
    }
}


// Iterators
impl <'a, T> Iterator for NdArrayIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            None => None,
            Some(index) => {
                let i = self.data.compute_index(&index);
                let ret = self.data.data().get(i).expect(&format!(
                    "Index out of bounds. Index: {:?}, array shape: {:?}, array strides: {:?}. array data len: {}",
                    index, self.data.shape(), self.data.strides(), self.data.data().len()
                ));

                Some(ret)
            }
        }
    }
}

impl <'a> Iterator for NdArrayDataIndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_done {
            return None;
        }

        let index = compute_index(&self.index, self.ref_data_strides);
        if index >= self.ref_data_len {
            panic!(
                "Index out of bounds. Index: {:?}, array shape: {:?}, array strides: {:?}. array data len: {}",
                self.index, self.ref_data_shape, self.ref_data_strides, self.ref_data_len
            )
        }
        let index = self.index.to_vec();

        self.index[self.axis_counter] += 1;

        let mut axis_change = false;
        while self.index[self.axis_counter] >= self.ref_data_shape[self.axis_counter] {
            axis_change = true;
            let len = self.index.len();
            self.index[self.axis_counter..len].iter_mut().for_each(|x| *x = 0);
            self.axis_counter = self.axis_counter.wrapping_sub(1);
            if self.axis_counter < self.index.len() {
                self.index[self.axis_counter] += 1;
            } else {
                self.has_done = true;
                break;
            }
        }

        if axis_change {
            self.axis_counter = self.index.len() - 1;
        }

        Some(index)
    }
}


/// # Example:
/// ```rust
/// use mnist_inference::NdArray;
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
                type IntoIter = NdArrayIterator<'b, T>;

                fn into_iter(self) -> Self::IntoIter {
                    NdArrayIterator::new(self)
                }
            }
        )+
    };
}

impl_nd_array_iter!{NdArray<T>, NdArrayView<'a, T>, NdArraySource<'a, T>}


// T conversions
impl <T: Clone> From<Vec<T>> for NdArray<T> {
    fn from(data: Vec<T>) -> Self {
        NdArray::new(data)
    }
}

impl <T> From<Scalar<T>> for NdArray<T> {
    fn from(data: Scalar<T>) -> Self {
        NdArray::new_shape_with_strides(Box::new([data.0]), vec![1], vec![1])
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
        Self::new_shape_with_strides(value.data.data().to_vec().into_boxed_slice(), value.shape, value.strides)
    }
}