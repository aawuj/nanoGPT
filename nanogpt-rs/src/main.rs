//! nanogpt-rs — a pure-Rust CPU reimplementation of nanoGPT.
//!
//! Subcommands:
//!   prepare  --input <input.txt> --out-dir <dir>      build train.bin/val.bin/meta.json
//!   train    [--data-dir ...] [--out-dir ...] [...]   train from scratch
//!   sample   [--checkpoint out-rs/ckpt.bin] [...]      generate text
//!
//! Run with `--help` after a subcommand for the full option list.

mod checkpoint;
mod data;
mod model;
mod ops;
mod optim;
mod pool;
mod rng;
mod sample;
mod train;

use std::path::Path;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        usage();
        return;
    }
    let opts = parse_opts(&argv[1..]);
    if opts.iter().any(|(k, _)| k == "help") {
        usage();
        return;
    }
    match argv[0].as_str() {
        "prepare" => cmd_prepare(&opts),
        "train" => cmd_train(&opts),
        "sample" => cmd_sample(&opts),
        other => {
            eprintln!("unknown subcommand: {}", other);
            usage();
            std::process::exit(1);
        }
    }
}

fn usage() {
    println!(
        "nanogpt-rs — nanoGPT reimplemented in Rust (CPU)\n\n\
         USAGE:\n  \
           nanogpt-rs prepare --input <input.txt> --out-dir <dir>\n  \
           nanogpt-rs train   [--data-dir D] [--out-dir O] [options]\n  \
           nanogpt-rs sample  [--checkpoint C] [options]\n\n\
         TRAIN OPTIONS (defaults tuned for shakespeare_char on CPU):\n  \
           --data-dir DIR        dataset dir with train.bin/val.bin (default: auto-detect data/shakespeare_char)\n  \
           --out-dir DIR         output dir (default out-rs)\n  \
           --block-size N        context length (64)\n  \
           --batch-size N        batch size (16)\n  \
           --n-layer N           transformer layers (4)\n  \
           --n-head N            attention heads (4)\n  \
           --n-embd N            embedding dim (128)\n  \
           --dropout P           dropout rate (0.2)\n  \
           --bias true|false     use biases in Linears/LayerNorms (false)\n  \
           --learning-rate F     max learning rate (1e-3)\n  \
           --max-iters N         training iterations (300)\n  \
           --weight-decay F      AdamW weight decay (0.1)\n  \
           --beta1 F / --beta2 F AdamW betas (0.9 / 0.99)\n  \
           --grad-clip F         global grad norm clip (1.0)\n  \
           --decay-lr true|false cosine LR decay (true)\n  \
           --warmup-iters N      linear warmup steps (50)\n  \
           --lr-decay-iters N    cosine decay horizon (max-iters)\n  \
           --min-lr F            minimum LR (1e-4)\n  \
           --eval-interval N     eval every N iters (50)\n  \
           --eval-iters N        batches per eval split (20)\n  \
           --log-interval N      log every N iters (10)\n  \
           --always-save true    save ckpt at every eval (false)\n  \
           --seed N              RNG seed (1337)\n  \
           --threads N           worker threads (auto)\n\n\
         SAMPLE OPTIONS:\n  \
           --checkpoint PATH     checkpoint file (out-rs/ckpt.bin)\n  \
           --start STR           prompt text, supports \\n escapes (\"\\n\")\n  \
           --num-samples N       number of samples (3)\n  \
           --max-new-tokens N    tokens per sample (300)\n  \
           --temperature F       sampling temperature (0.8)\n  \
           --top-k N             top-k filtering (40)\n  \
           --seed N              RNG seed (1337)\n  \
           --threads N           worker threads (auto)"
    );
}

/// Parse `--key value` pairs into a list (values may be missing for flags).
fn parse_opts(args: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(key) = a.strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                out.push((key.to_string(), args[i + 1].clone()));
                i += 2;
            } else {
                out.push((key.to_string(), "true".to_string()));
                i += 1;
            }
        } else {
            eprintln!("warning: ignoring unexpected argument '{}'", a);
            i += 1;
        }
    }
    out
}

fn get<'a>(opts: &'a [(String, String)], key: &str) -> Option<&'a str> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn get_parse<T: std::str::FromStr>(opts: &[(String, String)], key: &str) -> Option<T>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    get(opts, key).map(|v| v.parse::<T>().unwrap_or_else(|_| panic!("bad value for --{}: {}", key, v)))
}

fn get_bool(opts: &[(String, String)], key: &str, default: bool) -> bool {
    match get(opts, key) {
        Some(v) => matches!(v, "true" | "1" | "yes"),
        None => default,
    }
}

/// Detect the shakespeare_char data dir relative to the current working dir.
fn default_data_dir() -> String {
    for cand in [
        "data/shakespeare_char",
        "../data/shakespeare_char",
        "../../data/shakespeare_char",
    ] {
        if Path::new(cand).join("train.bin").exists() {
            return cand.to_string();
        }
    }
    "data/shakespeare_char".to_string()
}

fn cmd_prepare(opts: &[(String, String)]) {
    let input = get(opts, "input").unwrap_or_else(|| {
        eprintln!("error: prepare requires --input <input.txt>");
        std::process::exit(1);
    });
    let out_dir = get(opts, "out-dir").map(|s| s.to_string()).unwrap_or_else(|| {
        Path::new(input)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string())
    });
    match data::prepare(Path::new(input), Path::new(&out_dir)) {
        Ok((vocab, _itos)) => println!("done: vocab_size {}, wrote train.bin/val.bin/meta.json to {}", vocab, out_dir),
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_train(opts: &[(String, String)]) {
    let mut args = train::TrainArgs::default();
    args.data_dir = get(opts, "data-dir").map(|s| s.to_string()).unwrap_or_else(default_data_dir);
    if let Some(v) = get(opts, "out-dir") {
        args.out_dir = v.to_string();
    }
    if let Some(v) = get_parse(opts, "block-size") {
        args.block_size = v;
    }
    if let Some(v) = get_parse(opts, "batch-size") {
        args.batch_size = v;
    }
    if let Some(v) = get_parse(opts, "n-layer") {
        args.n_layer = v;
    }
    if let Some(v) = get_parse(opts, "n-head") {
        args.n_head = v;
    }
    if let Some(v) = get_parse(opts, "n-embd") {
        args.n_embd = v;
    }
    if let Some(v) = get_parse(opts, "dropout") {
        args.dropout = v;
    }
    args.bias = get_bool(opts, "bias", args.bias);
    if let Some(v) = get_parse(opts, "learning-rate") {
        args.learning_rate = v;
    }
    if let Some(v) = get_parse(opts, "max-iters") {
        args.max_iters = v;
        if get(opts, "lr-decay-iters").is_none() {
            args.lr_decay_iters = v; // ~= max_iters per Chinchilla, like train.py
        }
    }
    if let Some(v) = get_parse(opts, "weight-decay") {
        args.weight_decay = v;
    }
    if let Some(v) = get_parse(opts, "beta1") {
        args.beta1 = v;
    }
    if let Some(v) = get_parse(opts, "beta2") {
        args.beta2 = v;
    }
    if let Some(v) = get_parse(opts, "grad-clip") {
        args.grad_clip = v;
    }
    args.decay_lr = get_bool(opts, "decay-lr", args.decay_lr);
    if let Some(v) = get_parse(opts, "warmup-iters") {
        args.warmup_iters = v;
    }
    if let Some(v) = get_parse(opts, "lr-decay-iters") {
        args.lr_decay_iters = v;
    }
    if let Some(v) = get_parse(opts, "min-lr") {
        args.min_lr = v;
    }
    if let Some(v) = get_parse(opts, "eval-interval") {
        args.eval_interval = v;
    }
    if let Some(v) = get_parse(opts, "eval-iters") {
        args.eval_iters = v;
    }
    if let Some(v) = get_parse(opts, "log-interval") {
        args.log_interval = v;
    }
    args.always_save_checkpoint = get_bool(opts, "always-save", args.always_save_checkpoint);
    if let Some(v) = get_parse(opts, "seed") {
        args.seed = v;
    }
    if let Some(v) = get_parse(opts, "threads") {
        args.threads = v;
    }
    train::run(args);
}

fn cmd_sample(opts: &[(String, String)]) {
    let mut args = sample::SampleArgs::default();
    if let Some(v) = get(opts, "checkpoint") {
        args.checkpoint = v.to_string();
    }
    if let Some(v) = get(opts, "start") {
        args.start = v.to_string();
    }
    if let Some(v) = get_parse(opts, "num-samples") {
        args.num_samples = v;
    }
    if let Some(v) = get_parse(opts, "max-new-tokens") {
        args.max_new_tokens = v;
    }
    if let Some(v) = get_parse(opts, "temperature") {
        args.temperature = v;
    }
    if let Some(v) = get_parse(opts, "top-k") {
        args.top_k = v;
    }
    if let Some(v) = get_parse(opts, "seed") {
        args.seed = v;
    }
    if let Some(v) = get_parse(opts, "threads") {
        args.threads = v;
    }
    sample::run(args);
}
