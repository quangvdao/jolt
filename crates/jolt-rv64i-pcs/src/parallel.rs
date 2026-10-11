pub(crate) const MIN_TASK: usize = 4096;

pub(crate) fn enabled(len: usize) -> bool {
    len >= 2 * MIN_TASK && rayon::current_num_threads() > 1
}
