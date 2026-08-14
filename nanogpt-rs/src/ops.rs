//! Core numerical kernels: row-major matmul variants and softmax.
//! All matmul variants accumulate (`+=`) into `c`, so outputs must be
//! zeroed by the caller when a fresh result is wanted.
//!
//! Parallel safety: every kernel partitions its output so that concurrent
//! jobs write disjoint index ranges (see `ParMut` docs in pool.rs).

use crate::pool::{par_for, par_for_work, ParMut};

/// c[rows, cols] += a[rows, k] * b[k, cols]. Parallel over rows of c.
pub fn mm_nn(c: &mut [f32], a: &[f32], b: &[f32], rows: usize, k: usize, cols: usize, threads: usize) {
    let cp = ParMut::new(c);
    par_for_work(threads, rows, 2 * k * cols, |m| {
        let am = &a[m * k..(m + 1) * k];
        // SAFETY: each job owns exactly one disjoint row of c.
        let cm = unsafe { cp.slice(m * cols, (m + 1) * cols) };
        for i in 0..k {
            let av = am[i];
            if av == 0.0 {
                continue;
            }
            let brow = &b[i * cols..i * cols + cols];
            for o in 0..cols {
                cm[o] += av * brow[o];
            }
        }
    });
}

/// c[rows, cols] += a[rows, k] * b[cols, k]^T. Parallel over rows of c.
pub fn mm_nt(c: &mut [f32], a: &[f32], b: &[f32], rows: usize, k: usize, cols: usize, threads: usize) {
    let cp = ParMut::new(c);
    par_for_work(threads, rows, 2 * k * cols, |m| {
        let am = &a[m * k..(m + 1) * k];
        // SAFETY: each job owns exactly one disjoint row of c.
        let cm = unsafe { cp.slice(m * cols, (m + 1) * cols) };
        for o in 0..cols {
            let brow = &b[o * k..(o + 1) * k];
            let mut s = 0.0f32;
            for i in 0..k {
                s += am[i] * brow[i];
            }
            cm[o] += s;
        }
    });
}

/// c[k, cols] += a[rows, k]^T * b[rows, cols]. Parallel over output columns;
/// each job owns a disjoint column (strided but non-overlapping), so no
/// locking is needed.
pub fn mm_tn(c: &mut [f32], a: &[f32], b: &[f32], rows: usize, k: usize, cols: usize, threads: usize) {
    let cp = ParMut::new(c);
    par_for_work(threads, cols, 2 * rows * k, |o| {
        for m in 0..rows {
            let bv = b[m * cols + o];
            if bv == 0.0 {
                continue;
            }
            let am = &a[m * k..(m + 1) * k];
            for i in 0..k {
                // SAFETY: job owns column o exclusively.
                unsafe { cp.add_assign(i * cols + o, am[i] * bv) };
            }
        }
    });
}

/// x[rows, cols] += bias broadcast over rows.
pub fn add_bias_rows(x: &mut [f32], bias: &[f32], rows: usize, cols: usize, threads: usize) {
    let xp = ParMut::new(x);
    par_for_work(threads, rows, cols, |m| {
        // SAFETY: each job owns exactly one disjoint row of x.
        let xm = unsafe { xp.slice(m * cols, (m + 1) * cols) };
        for o in 0..cols {
            xm[o] += bias[o];
        }
    });
}

/// In-place numerically-stable softmax over a single row.
pub fn softmax_row(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}
