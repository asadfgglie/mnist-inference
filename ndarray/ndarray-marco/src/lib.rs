use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, LitInt};

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