use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Expr, LitInt, Token};

#[proc_macro]
pub fn nd_array_index(max_dim: TokenStream) -> TokenStream {
    let max_dim = parse_macro_input!(max_dim as LitInt);
    let max_dim: usize = max_dim.base10_parse().unwrap();

    let mut enum_variants = quote!();
    let mut from_slice_match_arms = quote!();
    let mut from_vec_match_arms = quote!();
    let mut deref_match_arms = quote!();
    let mut partial_eq_match_arms1 = quote!();
    let mut partial_eq_match_arms2 = quote!();
    let mut zeros_match_arms = quote!();
    let mut concat_match_arms = quote!();
    let mut index_macro_arms = quote!();

    let d = quote!($);

    for i in 0..=max_dim {
        let variant_name = format_ident!("Dim{}", i);

        enum_variants.extend(quote! {
            #variant_name([usize; #i]),
        });

        from_slice_match_arms.extend(quote! {
            #i => {
                let mut v = [0; #i];
                v.copy_from_slice(value);
                NdArrayIndex::#variant_name(v)
            }
        });

        from_vec_match_arms.extend(quote! {
            #i => {
                let mut v = [0; #i];
                v.copy_from_slice(value.as_slice());
                NdArrayIndex::#variant_name(v)
            }
        });

        deref_match_arms.extend(quote! {
            NdArrayIndex::#variant_name(v) => v,
        });

        partial_eq_match_arms1.extend(quote! {
            NdArrayIndex::#variant_name(v) => v == other,
        });

        partial_eq_match_arms2.extend(quote! {
            NdArrayIndex::#variant_name(v) => v == self,
        });

        zeros_match_arms.extend(quote! {
            #i => NdArrayIndex::#variant_name([0; #i]),
        });

        concat_match_arms.extend(quote! {
            #i => {
                let mut v = [0; #i];
                let (v1, v2) = v.split_at_mut(self.len());
                v1.copy_from_slice(&self);
                v2.copy_from_slice(&other);
                NdArrayIndex::#variant_name(v)
            },
        });

        let idents: Vec<_> = (1..=i).map(|n| format_ident!("x{}", n)).collect();
        index_macro_arms.extend(quote! {
            ( #( #d #idents:expr ),* $(,)? ) => {
                $crate::NdArrayIndex::#variant_name([ #( #d #idents ),* ])
            };
        });
    }

    TokenStream::from(quote! {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum NdArrayIndex {
            #enum_variants
            DyDim(Vec<usize>),
        }

        impl From<&[usize]> for NdArrayIndex {
            fn from(value: &[usize]) -> Self {
                match value.len() {
                    #from_slice_match_arms
                    _ => NdArrayIndex::DyDim(value.to_vec()),
                }
            }
        }

        impl From<Vec<usize>> for NdArrayIndex {
                fn from(value: Vec<usize>) -> Self {
                    match value.len() {
                        #from_vec_match_arms
                        _ => NdArrayIndex::DyDim(value),
                    }
                }
            }

        impl std::ops::Deref for NdArrayIndex {
            type Target = [usize];
            fn deref(&self) -> &Self::Target {
                match self {
                    #deref_match_arms
                    NdArrayIndex::DyDim(v) => v,
                }
            }
        }

        impl DerefMut for NdArrayIndex {
            fn deref_mut(&mut self) -> &mut Self::Target {
                match self {
                    #deref_match_arms
                    NdArrayIndex::DyDim(v) => v,
                }
            }
        }

        impl PartialEq<&[usize]> for NdArrayIndex {
            fn eq(&self, other: &&[usize]) -> bool {
                if self.len() != other.len() {
                    false
                } else {
                    match self {
                        #partial_eq_match_arms1
                        NdArrayIndex::DyDim(v) => v == other,
                    }
                }
            }
        }

        impl PartialEq<NdArrayIndex> for &[usize] {
            fn eq(&self, other: &NdArrayIndex) -> bool {
                if self.len() != other.len() {
                    false
                } else {
                    match other {
                        #partial_eq_match_arms2
                        NdArrayIndex::DyDim(v) => v == self,
                    }
                }
            }
        }

        impl NdArrayIndex {
            pub fn zeros(len: usize) -> Self {
                match len {
                    #zeros_match_arms
                    _ => NdArrayIndex::DyDim(vec![0; len]),
                }
            }
            pub fn concat(self, other: NdArrayIndex) -> Self {
                match self.len() + other.len() {
                    #concat_match_arms
                    _ =>  {
                        let mut v = Vec::with_capacity(self.len() + other.len());
                        v.extend(self.into_iter());
                        v.extend(other.into_iter());
                        v.into()
                    }
                }
            }
        }

        #[macro_export]
        macro_rules! index {
            #index_macro_arms
            ( $( $x:expr ),+ $(,)? ) => {
                $crate::NdArrayIndex::DyDim(vec![$($x),+])
            };
        }
    })
}

struct UsizeExpr(pub Expr);

impl Parse for UsizeExpr {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(UsizeExpr(input.parse()?))
    }
}

impl ToTokens for UsizeExpr {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.0.to_tokens(tokens)
    }
}

enum Slice {                                                            // py means    rust means
    Step { step: UsizeExpr },                                           // ::2      or (..).step_by(2)
    All,                                                                // :        or ..
    RangeTo { end: UsizeExpr },                                         // :3       or ..3
    RangeToStep { end: UsizeExpr, step: UsizeExpr },                    // :3:2     or (..3).step_by(2)
    Index { index: UsizeExpr },                                         // 0
    RangeFromStep { start: UsizeExpr, step: UsizeExpr },                // 1::2     or (1..).step_by(2)
    RangeFrom { start: UsizeExpr },                                     // 1:       or 1..
    Range { start: UsizeExpr, end: UsizeExpr },                         // 1:3      or 1..3
    RangeStep { start: UsizeExpr, end: UsizeExpr, step: UsizeExpr },    // 1:10:2   or (1..10).step_by(2)
}

struct Slices(pub Punctuated<Slice, Token![,]>);

impl Parse for Slice {
    /// ```cfg
    /// slice ::= "::" `usize expr` |                               // Step
    ///           ":" |                                             // All
    ///           ":" `usize expr` |                                // RangeTo
    ///           ":" `usize expr` ":" `usize expr` |               // RangeToStep
    ///           `usize expr` |                                    // Index
    ///           `usize expr` "::" `usize expr`;                   // RangeFromStep
    ///           `usize expr` ":" |                                // RangeFrom
    ///           `usize expr` ":" `usize expr` |                   // Range
    ///           `usize expr` ":" `usize expr` ":" `usize expr`    // RangeStep
    /// ```
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![::]) {
            input.parse::<Token![::]>()?;
            Ok(Slice::Step { step: input.parse::<UsizeExpr>()?  })
        } else if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;

            if !input.is_empty() && !input.peek(Token![,]) {
                let end = input.parse::<UsizeExpr>()?;

                if input.peek(Token![:]) {
                    input.parse::<Token![:]>()?;

                    let step = input.parse::<UsizeExpr>()?;
                    Ok(Slice::RangeToStep { end, step })
                } else {
                    Ok(Slice::RangeTo { end })
                }
            } else {
                Ok(Slice::All)
            }
        } else if !input.is_empty() && !input.peek(Token![,]) {
            let index = input.parse::<UsizeExpr>()?;

            if input.peek(Token![::]) {
                input.parse::<Token![::]>()?;

                let start = index;
                let step = input.parse::<UsizeExpr>()?;
                Ok(Slice::RangeFromStep { start, step })

            } else if input.peek(Token![:]) {
                input.parse::<Token![:]>()?;

                let start = index;
                if !input.is_empty() && !input.peek(Token![,]) {
                    let end = input.parse::<UsizeExpr>()?;

                    if input.peek(Token![:]) {
                        input.parse::<Token![:]>()?;

                        let step = input.parse::<UsizeExpr>()?;
                        Ok(Slice::RangeStep { start, end, step })
                    } else {
                        Ok(Slice::Range { start, end })
                    }
                } else {
                    Ok(Slice::RangeFrom { start })
                }
            } else {
                Ok(Slice::Index { index })
            }
        } else {
            Err(input.error("Invalid slice syntax"))
        }
    }
}

impl Parse for Slices {
    fn parse(input: ParseStream) -> Result<Self> {
        let slices = input.parse_terminated(Slice::parse, Token![,])?;
        Ok(Slices(slices))
    }
}


/// use python slice syntax
/// ```cfg
/// slices ::= (slice | slice "," slices) ","{0,1}
/// slice ::= ":" |                                             // All
///           ":" `usize expr` |                                // RangeTo
///           ":" `usize expr` ":" `usize expr` |               // RangeToStep
///           "::" `usize expr` |                               // Step
///           `usize expr` |                                    // Index
///           `usize expr` ":" |                                // RangeFrom
///           `usize expr` ":" `usize expr` |                   // Range
///           `usize expr` ":" `usize expr` ":" `usize expr`    // RangeStep
///           `usize expr` "::" `usize expr`;                   // RangeFromStep
/// ```
///
/// # Example
/// ```ignore
/// use ndarray::{slice, AxisSlice};
/// assert_eq!(slice![:], [AxisSlice::All]);
///
/// assert_eq!(slice![1+1,], [AxisSlice::Index { index: 1+1 }]);
///
/// assert_eq!(slice![1:3+9*2], [AxisSlice::Range { start: 1, end: 3+9*2 }]);
///
/// assert_eq!(slice![1:], [AxisSlice::RangeFrom { start: 1 }]);
///
/// assert_eq!(slice![1::2], [AxisSlice::RangeFromStep { start: 1, step: 2 }]);
///
/// assert_eq!(slice![:3], [AxisSlice::RangeTo { end: 3 }]);
///
/// assert_eq!(slice![:3:2], [AxisSlice::RangeToStep { end: 3, step: 2 }]);
///
/// assert_eq!(slice![1:10:2], [AxisSlice::RangeStep { start: 1, end: 10, step: 2 }]);
///
/// assert_eq!(slice![::2], [AxisSlice::Step { step: 2 }]);
/// ```
#[proc_macro]
pub fn slice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Slices);
    let idents: Vec<_> = input.0.into_iter()
        .map(|s| match s {
            Slice::All => {quote! {
                ::ndarray::axis::AxisSlice::All
            }},
            Slice::Index { index } => {quote! {
                ::ndarray::axis::AxisSlice::Index{ index: #index }
            }},
            Slice::Range { start, end } => {quote! {
                ::ndarray::axis::AxisSlice::Range { start: #start, end: #end }
            }},
            Slice::RangeFrom { start } => {quote! {
                ::ndarray::axis::AxisSlice::RangeFrom { start: #start }
            }},
            Slice::RangeFromStep { start, step } => {quote! {
                ::ndarray::axis::AxisSlice::RangeFromStep { start: #start, step: #step }
            }},
            Slice::RangeTo { end } => {quote! {
                ::ndarray::axis::AxisSlice::RangeTo { end: #end }
            }},
            Slice::RangeToStep { end, step } => {quote! {
                ::ndarray::axis::AxisSlice::RangeToStep { end: #end, step: #step }
            }},
            Slice::RangeStep { start, end, step } => {quote! {
                ::ndarray::axis::AxisSlice::RangeStep { start: #start, end: #end, step: #step }
            }},
            Slice::Step { step } => {quote! {
                ::ndarray::axis::AxisSlice::Step{ step: #step }
            }},
        })
        .collect();

    TokenStream::from(quote! {
        [ #( #idents ),* ]
    })
}