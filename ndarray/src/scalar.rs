use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use crate::array::NdArray;
use crate::{scalar_assign_op, scalar_op, Cast, NdArrayIndex};

#[derive(Debug, PartialEq, Clone)]
pub struct Scalar<T>(pub T);

impl <T: Copy> Copy for Scalar<T> {}

impl <T> From<Scalar<T>> for NdArray<T> {
    fn from(data: Scalar<T>) -> Self {
        NdArray::new_array(Box::new([data.0]), NdArrayIndex::Dim1([1]), NdArrayIndex::Dim1([1]), 0)
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

impl <T, NT> Cast<NT> for Scalar<T>
where T: Into<NT> {
    type Target = Scalar<NT>;

    fn cast(self) -> Self::Target {
        Scalar(self.0.into())
    }
}

scalar_op!{(+, Add, add), (-, Sub, sub), (*, Mul, mul), (/, Div, div), (%, Rem, rem)}
scalar_assign_op!{
    (+=, AddAssign, add_assign), (-=, SubAssign, sub_assign), (*=, MulAssign, mul_assign),
    (/=, DivAssign, div_assign), (%=, RemAssign, rem_assign)
}