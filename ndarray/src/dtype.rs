use crate::HasDtype;
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_complex::Complex32;
use safetensors::Dtype;

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(bool::DTYPE.bitsize(), size_of::<bool>() * 8)
/// ```
impl HasDtype for bool {
    const DTYPE: Dtype = Dtype::BOOL;
}

// impl HasDtype for f4 {
//     const DTYPE: Dtype = Dtype::F4;
// }
//
// impl HasDtype for f6_e2m3 {
//     const DTYPE: Dtype = Dtype::F6_E2M3;
// }
//
// impl HasDtype for f6_e3m2 {
//     const DTYPE: Dtype = Dtype::F6_E3M2;
// }

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(u8::DTYPE.bitsize(), size_of::<u8>() * 8)
/// ```
impl HasDtype for u8 {
    const DTYPE: Dtype = Dtype::U8;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(i8::DTYPE.bitsize(), size_of::<i8>() * 8)
/// ```
impl HasDtype for i8 {
    const DTYPE: Dtype = Dtype::I8;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
/// use float8::F8E5M2;
///
/// assert_eq!(F8E5M2::DTYPE.bitsize(), size_of::<F8E5M2>() * 8)
/// ```
impl HasDtype for F8E5M2 {
    const DTYPE: Dtype = Dtype::F8_E5M2;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
/// use float8::F8E4M3;
///
/// assert_eq!(F8E4M3::DTYPE.bitsize(), size_of::<F8E4M3>() * 8)
/// ```
impl HasDtype for F8E4M3 {
    const DTYPE: Dtype = Dtype::F8_E4M3;
}

// impl HasDtype for f8_e8m0 {
//     const DTYPE: Dtype = Dtype::F8_E8M0;
// }

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(i16::DTYPE.bitsize(), size_of::<i16>() * 8)
/// ```
impl HasDtype for i16 {
    const DTYPE: Dtype = Dtype::I16;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(u16::DTYPE.bitsize(), size_of::<u16>() * 8)
/// ```
impl HasDtype for u16 {
    const DTYPE: Dtype = Dtype::U16;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
/// use half::f16;
///
/// assert_eq!(f16::DTYPE.bitsize(), size_of::<f16>() * 8)
/// ```
impl HasDtype for f16 {
    const DTYPE: Dtype = Dtype::F16;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
/// use half::bf16;
///
/// assert_eq!(bf16::DTYPE.bitsize(), size_of::<bf16>() * 8)
/// ```
impl HasDtype for bf16 {
    const DTYPE: Dtype = Dtype::BF16;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(i32::DTYPE.bitsize(), size_of::<i32>() * 8)
/// ```
impl HasDtype for i32 {
    const DTYPE: Dtype = Dtype::I32;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(u32::DTYPE.bitsize(), size_of::<u32>() * 8)
/// ```
impl HasDtype for u32 {
    const DTYPE: Dtype = Dtype::U32;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(f32::DTYPE.bitsize(), size_of::<f32>() * 8)
/// ```
impl HasDtype for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
/// use num_complex::Complex32;
///
/// assert_eq!(Complex32::DTYPE.bitsize(), size_of::<Complex32>() * 8)
/// ```
impl HasDtype for Complex32 {
    const DTYPE: Dtype = Dtype::C64;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(f64::DTYPE.bitsize(), size_of::<f64>() * 8)
/// ```
impl HasDtype for f64 {
    const DTYPE: Dtype = Dtype::F64;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(i64::DTYPE.bitsize(), size_of::<i64>() * 8)
/// ```
impl HasDtype for i64 {
    const DTYPE: Dtype = Dtype::I64;
}

/// ```
/// use ndarray::HasDtype;
/// use safetensors::Dtype;
///
/// assert_eq!(u64::DTYPE.bitsize(), size_of::<u64>() * 8)
/// ```
impl HasDtype for u64 {
    const DTYPE: Dtype = Dtype::U64;
}
