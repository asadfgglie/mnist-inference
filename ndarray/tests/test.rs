use ndarray::{AxisSlice, slice};

#[test]
fn test_slice_marco() {
    assert_eq!(slice![:], [AxisSlice::All]);

    assert_eq!(slice![1 + 1,], [AxisSlice::Index { index: 1 + 1 }]);

    assert_eq!(
        slice![1:3+9*2],
        [AxisSlice::Range {
            start: 1,
            end: 3 + 9 * 2
        }]
    );

    assert_eq!(slice![1:], [AxisSlice::RangeFrom { start: 1 }]);

    assert_eq!(
        slice![1::2],
        [AxisSlice::RangeFromStep { start: 1, step: 2 }]
    );

    assert_eq!(slice![:3], [AxisSlice::RangeTo { end: 3 }]);

    assert_eq!(slice![:3:2], [AxisSlice::RangeToStep { end: 3, step: 2 }]);

    assert_eq!(
        slice![1:10:2],
        [AxisSlice::RangeStep {
            start: 1,
            end: 10,
            step: 2
        }]
    );

    assert_eq!(slice![::2], [AxisSlice::Step { step: 2 }]);
}
