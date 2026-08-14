//! Sampling from a trained checkpoint — Rust port of `sample.py`.

use crate::checkpoint::load_checkpoint;
use crate::model::Gpt;
use crate::rng::Rng;
use std::collections::HashMap;
use std::path::Path;

pub struct SampleArgs {
    pub checkpoint: String,
    pub start: String,
    pub num_samples: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
    pub threads: usize,
}

impl Default for SampleArgs {
    fn default() -> Self {
        SampleArgs {
            checkpoint: "out-rs/ckpt.bin".to_string(),
            start: "\n".to_string(),
            num_samples: 3,
            max_new_tokens: 300,
            temperature: 0.8,
            top_k: 40,
            seed: 1337,
            threads: crate::pool::default_threads(),
        }
    }
}

pub fn run(args: SampleArgs) {
    let ckpt = match load_checkpoint(Path::new(&args.checkpoint)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load checkpoint '{}': {}", args.checkpoint, e);
            std::process::exit(1);
        }
    };
    println!(
        "loaded checkpoint {} (iter {}, best val loss {:.4})",
        args.checkpoint, ckpt.iter_num, ckpt.best_val_loss
    );
    let model: Gpt = crate::checkpoint::model_from_checkpoint(&ckpt).expect("invalid checkpoint");

    let itos = ckpt.itos.clone();
    let mut stoi = HashMap::new();
    for (i, c) in itos.iter().enumerate() {
        if let Some(ch) = c.chars().next() {
            stoi.insert(ch, i as u32);
        }
    }

    // encode the prompt (supports \n \r \t escapes)
    let start = unescape(&args.start);
    let mut start_ids: Vec<u32> = Vec::new();
    for ch in start.chars() {
        match stoi.get(&ch) {
            Some(&id) => start_ids.push(id),
            None => {
                eprintln!("warning: prompt char {:?} not in vocabulary, skipping", ch);
            }
        }
    }
    if start_ids.is_empty() {
        start_ids.push(0);
    }

    let mut rng = Rng::new(args.seed);
    for k in 0..args.num_samples {
        let ids = model.generate(
            &start_ids,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            &mut rng,
            args.threads,
        );
        println!("----- sample {} -----", k);
        if itos.is_empty() {
            // no decoder available: print raw token ids
            println!("{:?}", ids);
        } else {
            let text: String = ids
                .iter()
                .map(|&id| itos.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
                .collect();
            println!("{}", text);
        }
    }
}

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}
