//! Training loop — Rust port of nanoGPT's `train.py` (single-process, CPU).

use crate::checkpoint::save_checkpoint;
use crate::data::{Dataset, Split};
use crate::model::{Cache, Gpt, GptConfig};
use crate::optim::{clip_grad_norm, get_lr, AdamW};
use crate::rng::Rng;
use std::path::Path;
use std::time::Instant;

pub struct TrainArgs {
    pub data_dir: String,
    pub out_dir: String,
    pub block_size: usize,
    pub batch_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub bias: bool,
    pub learning_rate: f32,
    pub max_iters: usize,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub grad_clip: f32,
    pub decay_lr: bool,
    pub warmup_iters: usize,
    pub lr_decay_iters: usize,
    pub min_lr: f32,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub log_interval: usize,
    pub always_save_checkpoint: bool,
    pub seed: u64,
    pub threads: usize,
}

impl Default for TrainArgs {
    /// Defaults follow config/train_shakespeare_char.py, scaled down so a
    /// CPU run converges in a few minutes.
    fn default() -> Self {
        TrainArgs {
            data_dir: String::new(),
            out_dir: "out-rs".to_string(),
            block_size: 64,
            batch_size: 16,
            n_layer: 4,
            n_head: 4,
            n_embd: 128,
            dropout: 0.2,
            bias: false,
            learning_rate: 1e-3,
            max_iters: 300,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            grad_clip: 1.0,
            decay_lr: true,
            warmup_iters: 50,
            lr_decay_iters: 300,
            min_lr: 1e-4,
            eval_interval: 50,
            eval_iters: 20,
            log_interval: 10,
            always_save_checkpoint: false,
            seed: 1337,
            threads: crate::pool::default_threads(),
        }
    }
}

pub fn run(args: TrainArgs) {
    let dataset = match Dataset::load(&args.data_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "error: failed to load dataset from '{}': {}\n\
                 hint: run `nanogpt-rs prepare` or `python data/shakespeare_char/prepare.py` first",
                args.data_dir, e
            );
            std::process::exit(1);
        }
    };
    println!(
        "dataset loaded: train {} tokens, val {} tokens, vocab_size {}{}",
        dataset.train.len(),
        dataset.val.len(),
        dataset.vocab_size,
        if dataset.itos.is_empty() { " (itos unknown)" } else { "" }
    );

    let mut rng = Rng::new(args.seed);
    let cfg = GptConfig {
        block_size: args.block_size,
        vocab_size: dataset.vocab_size,
        n_layer: args.n_layer,
        n_head: args.n_head,
        n_embd: args.n_embd,
        dropout: args.dropout,
        bias: args.bias,
    };
    let mut model = Gpt::new(cfg, &mut rng);
    println!("number of parameters: {:.2}M", model.num_params() as f64 / 1e6);
    let n_decay: usize = model.decay.iter().filter(|&&d| d).count();
    println!(
        "num decayed parameter elements: {} ({:.2}M), non-decayed: {}",
        n_decay,
        n_decay as f64 / 1e6,
        model.params.len() - n_decay
    );

    let mut optimizer = AdamW::new(model.params.len(), args.beta1, args.beta2, 1e-8);
    let mut cache = Cache::default();
    let mut eval_cache = Cache::default();
    let mut x: Vec<u32> = Vec::new();
    let mut y: Vec<u32> = Vec::new();

    // poor man's data loader: fetch the very first batch
    dataset.get_batch(Split::Train, args.batch_size, args.block_size, &mut rng, &mut x, &mut y);

    let mut best_val_loss = 1e9f64;
    let t_start = Instant::now();
    let mut t0 = Instant::now();

    for it in 0..=args.max_iters {
        // determine and set the learning rate for this iteration
        let lr = if args.decay_lr {
            get_lr(it, args.warmup_iters, args.lr_decay_iters, args.learning_rate, args.min_lr)
        } else {
            args.learning_rate
        };

        // evaluate the loss on train/val sets and write checkpoints
        if it % args.eval_interval == 0 {
            let (train_loss, val_loss) = estimate_loss(&model, &dataset, &args, &mut rng, &mut eval_cache);
            println!("step {}: train loss {:.4}, val loss {:.4}", it, train_loss, val_loss);
            if val_loss < best_val_loss || args.always_save_checkpoint {
                best_val_loss = val_loss;
                if it > 0 {
                    let ckpt_path = Path::new(&args.out_dir).join("ckpt.bin");
                    save_checkpoint(&ckpt_path, &model, it as u64, best_val_loss, &dataset.itos)
                        .expect("failed to save checkpoint");
                    println!("saving checkpoint to {}", ckpt_path.display());
                }
            }
        }

        // forward + backward + update
        model.zero_grad();
        let loss = model.forward(&x, Some(&y), true, &mut rng, &mut cache, args.threads);
        model.backward(&mut cache, &y, args.threads);
        // immediately prefetch next batch (like train.py's async prefetch)
        dataset.get_batch(Split::Train, args.batch_size, args.block_size, &mut rng, &mut x, &mut y);
        clip_grad_norm(&mut model.grads, args.grad_clip);
        optimizer.step(&mut model.params, &model.grads, &model.decay, lr, args.weight_decay);

        if it % args.log_interval == 0 {
            let dt = t0.elapsed();
            t0 = Instant::now();
            println!("iter {}: loss {:.4}, lr {:.6}, time {:.2}ms", it, loss, lr, dt.as_secs_f64() * 1000.0);
        }
    }

    // final evaluation + unconditional final checkpoint
    let (train_loss, val_loss) = estimate_loss(&model, &dataset, &args, &mut rng, &mut eval_cache);
    println!(
        "final: train loss {:.4}, val loss {:.4} (best val {:.4}), total time {:.1}s",
        train_loss,
        val_loss,
        best_val_loss.min(val_loss),
        t_start.elapsed().as_secs_f64()
    );
    let ckpt_path = Path::new(&args.out_dir).join("ckpt.bin");
    save_checkpoint(&ckpt_path, &model, args.max_iters as u64, best_val_loss.min(val_loss), &dataset.itos)
        .expect("failed to save checkpoint");
    println!("saved final checkpoint to {}", ckpt_path.display());
}

/// Mirrors `estimate_loss` from train.py: mean loss over `eval_iters`
/// random batches of each split, with dropout disabled.
fn estimate_loss(
    model: &Gpt, dataset: &Dataset, args: &TrainArgs, rng: &mut Rng, cache: &mut Cache,
) -> (f64, f64) {
    let mut x: Vec<u32> = Vec::new();
    let mut y: Vec<u32> = Vec::new();
    let mut out = [0.0f64; 2];
    for (si, split) in [Split::Train, Split::Val].iter().enumerate() {
        let mut sum = 0.0f64;
        for _ in 0..args.eval_iters {
            dataset.get_batch(*split, args.batch_size, args.block_size, rng, &mut x, &mut y);
            let loss = model.forward(&x, Some(&y), false, rng, cache, args.threads);
            sum += loss as f64;
        }
        out[si] = sum / args.eval_iters as f64;
    }
    (out[0], out[1])
}
