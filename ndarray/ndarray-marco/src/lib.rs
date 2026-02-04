use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, LitInt};

#[proc_macro]
pub fn nd_array_index(max_dim: TokenStream) -> TokenStream {
    let max_dim = parse_macro_input!(max_dim as LitInt);
    let max_dim: usize = max_dim.base10_parse().unwrap();

    let mut enum_variants = quote! {};
    let mut from_slice_match_arms = quote! {};
    let mut from_vec_match_arms = quote! {};
    let mut deref_match_arms = quote! {};
    let mut zeros_match_arms = quote! {};
    let mut index_macro_arms = quote! {};

    let d = quote!( $ );

    for i in 1..=max_dim {
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

        zeros_match_arms.extend(quote! {
            #i => NdArrayIndex::#variant_name([0; #i]),
        });

        let idents: Vec<_> = (1..=i).map(|n| format_ident!("x{}", n)).collect();
        index_macro_arms.extend(quote! {
            ( #( #d #idents:expr ),* $(,)? ) => {
                NdArrayIndex::#variant_name([ #( #d #idents ),* ])
            };
        });
    }

    TokenStream::from(quote! {
        #[derive(Debug, Clone, PartialEq)]
        pub enum NdArrayIndex {
            #enum_variants
            DyDim(Vec<usize>),
        }

        impl From<&[usize]> for NdArrayIndex {
            fn from(value: &[usize]) -> Self {
                match value.len() {
                    0 => panic!("Zero sized index not allowed"),
                    #from_slice_match_arms
                    _ => NdArrayIndex::DyDim(value.to_vec()),
                }
            }
        }

        impl From<Vec<usize>> for NdArrayIndex {
                fn from(value: Vec<usize>) -> Self {
                    match value.len() {
                        0 => panic!("zero sized index is not allowed."),
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

        impl NdArrayIndex {
            pub fn zeros(len: usize) -> Self {
                match len {
                    0 => panic!("zero sized index is not allowed."),
                    #zeros_match_arms
                    _ => NdArrayIndex::DyDim(vec![0; len]),
                }
            }
        }

        macro_rules! index {
            #index_macro_arms
            ( $( $x:expr ),+ $(,)? ) => {
                NdArrayIndex::DyDim(vec![$($x),+])
            };
        }
    })
}