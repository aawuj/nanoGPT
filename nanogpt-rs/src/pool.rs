//! Lightweight scoped data-parallelism helper (std::thread::scope based).

/// Run `f(i)` for `i in 0..n` across at most `threads` OS threads.
/// The current thread participates in the work as well.
///
/// `work_per_item` is a rough per-item cost in flops, used to decide
/// whether spawning threads is worth it at all.
///
/// `f` must be `Fn` (not `FnMut`): parallel jobs cannot own mutable state.
/// To write into shared buffers from jobs, capture a [`ParMut`] wrapper and
/// guarantee that the index ranges touched by different jobs are disjoint.
pub fn par_for_work(threads: usize, n: usize, work_per_item: usize, f: impl Fn(usize) + Sync) {
    if n == 0 {
        return;
    }
    // Only parallelize when the total workload justifies thread spawn/join.
    let total_work = n.saturating_mul(work_per_item);
    let nt = if total_work < 1 << 20 { 1 } else { threads.min(n).max(1) };
    if nt == 1 {
        for i in 0..n {
            f(i);
        }
        return;
    }
    let chunk = n.div_ceil(nt);
    let fref = &f;
    std::thread::scope(|s| {
        let mut start = chunk;
        while start < n {
            let end = (start + chunk).min(n);
            s.spawn(move || {
                for i in start..end {
                    fref(i);
                }
            });
            start += chunk;
        }
        // chunk 0 runs on the calling thread
        for i in 0..chunk.min(n) {
            f(i);
        }
    });
}

/// Convenience form assuming each item is a heavyweight job.
pub fn par_for(threads: usize, n: usize, f: impl Fn(usize) + Sync) {
    par_for_work(threads, n, usize::MAX / 2, f);
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Copyable raw-pointer view over a mutable slice, usable from `Fn`
/// closures running on multiple threads.
///
/// SAFETY contract: every kernel in this crate partitions work so that
/// concurrent jobs touch *disjoint* indices of each `ParMut` buffer.
#[derive(Clone, Copy)]
pub struct ParMut<T> {
    p: *mut T,
    n: usize,
}

unsafe impl<T: Send> Sync for ParMut<T> {}

impl<T> ParMut<T> {
    pub fn new(s: &mut [T]) -> Self {
        ParMut { p: s.as_mut_ptr(), n: s.len() }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// SAFETY: concurrent jobs must not overlap on the returned range.
    #[inline]
    pub unsafe fn slice(&self, start: usize, end: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.p.add(start), end - start)
    }

    #[inline]
    pub unsafe fn get(&self, i: usize) -> T
    where
        T: Copy,
    {
        *self.p.add(i)
    }

    #[inline]
    pub unsafe fn set(&self, i: usize, v: T) {
        *self.p.add(i) = v;
    }

    #[inline]
    pub unsafe fn add_assign(&self, i: usize, v: T)
    where
        T: std::ops::AddAssign,
    {
        *self.p.add(i) += v;
    }
}
