pub use ndarray_marco::slice;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AxisSlice {                                        // py means    rust means
    All,                                                    // :        or ..
    Index { index: usize },                                 // 0
    Range { start: usize, end: usize },                     // 1:3      or 1..3
    RangeFrom { start: usize },                             // 1:       or 1..
    RangeFromStep { start: usize, step: usize },            // 1::2     or (1..).step_by(2)
    RangeTo { end: usize },                                 // :3       or ..3
    RangeToStep { end: usize, step: usize },                // :3:2     or (..3).step_by(2)
    RangeStep { start: usize, end: usize, step: usize },    // 1:10:2   or (1..10).step_by(2)
    Step { step: usize },                                   // ::2      or (..).step_by(2)
}