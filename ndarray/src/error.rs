#[derive(Debug, PartialEq)]
pub enum NdArrayError {
    BroadcastError(String),
    ReshapeError(String),
    PermuteError(String),
    SliceError(String),
    IncompatibleReshapeError(String),
    InvalidStridesError(String),
    InvalidShapeError(String),
    DtypeMismatch(String),
    InvalidBufferSize(String),
    Misaligned(String),
}