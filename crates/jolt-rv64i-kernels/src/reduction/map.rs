/// A source range in the 256 committed columns. Indicator digit zero and absence
/// both encode zero; digit `k >= 1` selects column `start + k - 1`.
/// Flags span one column per zero-bit digit and encode its presence, in list order.
/// Flag groups contain one through eight source columns.
#[derive(Clone, Debug)]
pub enum ColumnMap {
    Word { start: usize, trace_word: usize },
    Indicators { start: usize, column: usize },
    Flags { start: usize, columns: Vec<usize> },
}
