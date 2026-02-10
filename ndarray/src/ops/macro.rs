// Scalar math op
#[macro_export]
macro_rules! scalar_op {
    ($( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+) => {
        $(
            impl <L, R> $op_trait<Scalar<R>> for Scalar<L>
            where L: $op_trait<Output=L>, R: Into<L> {
                type Output = Self;

                fn $op_fn(self, rhs: Scalar<R>) -> Self::Output {
                    Self(self.0 $op rhs.0.into())
                }
            }
        )+
    };
}

#[macro_export]
macro_rules! scalar_assign_op {
    ($( ( $op:tt, $op_trait:ident, $op_fn:ident ) ),+) => {
        $(
            impl <L, R> $op_trait<Scalar<R>> for Scalar<L>
            where L: $op_trait, R: Into<L> {
                fn $op_fn(&mut self, rhs: Scalar<R>) {
                    self.0 $op rhs.0.into()
                }
            }
            impl <L, R> $op_trait<&Scalar<R>> for Scalar<L>
            where L: $op_trait, R: Into<L> + Clone {
                fn $op_fn(&mut self, rhs: &Scalar<R>) {
                    let rhs: Scalar<R> = rhs.clone();
                    *self $op rhs;
                }
            }
        )+
    };
}

/// basic `&NdArray`/`NdArray` operation with `&NdArray`/`NdArray`
#[macro_export]
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
#[macro_export]
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
#[macro_export]
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

                    Self::Output::new_shape(data, lhs.shape)
                }
            }
        )+
    };
}

/// combine `op!`, `ref_view_op!`, `ref_op!`
///
/// auto implement `NdArrayView` marco
#[macro_export]
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
///
/// let mut a: NdArray<f32> = NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast();
/// a += NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]);
/// assert_eq!(a, NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast() * Scalar(2i8));
///
/// a /= Scalar(2i8);
/// assert_eq!(a, NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]).cast());
///
/// a -= NdArray::new_shape(Vec::from_iter((1i8..10i8).into_iter()), vec![3,3]);
/// assert_eq!(a, NdArray::new_shape(vec![0.0; 9], vec![3,3]));
/// ```
#[macro_export]
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

                    for indices in self.iter_index().collect::<Vec<_>>() {
                        self[indices.clone()] $op rhs[indices].clone().into();
                    }
                }
            }
        )+
    };
}

/// basic `&NdArrayLike` assign (in-place) operation with `NdArray`
#[macro_export]
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

                    for indices in self.iter_index().collect::<Vec<_>>() {
                        self[indices.clone()] $op rhs[indices].clone().into();
                    }
                }
            }
        )+
    };
}

/// combine `assign_op!`, `ref_assign_op!`
///
/// auto implement `NdArrayView` marco
#[macro_export]
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
#[macro_export]
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
#[macro_export]
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
#[macro_export]
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
#[macro_export]
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
#[macro_export]
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
#[macro_export]
macro_rules! general_eco_op_scalar {
    ($op:tt, $op_trait:ident, $op_fn:ident) => {
        eco_op_scalar! {$op, $op_trait, $op_fn}
        ref_eco_op_scalar! {(NdArrayView), [($op, $op_trait, $op_fn)]}
    };
}

/// base on vector space axiom and language meanings, we only implement `*`, `/`, and `%`
#[macro_export]
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
