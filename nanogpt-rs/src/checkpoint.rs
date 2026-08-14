//! Checkpoint save/load in a small self-describing binary format:
//!
//! magic "NPGPTCK1" | version u32 | config | iter u64 | best_val_loss f64
//! | itos (char strings) | param count u64 | raw f32 (LE) weights

use crate::model::{Gpt, GptConfig};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"NPGPTCK1";
const VERSION: u32 = 1;

pub struct Checkpoint {
    pub cfg: GptConfig,
    pub params: Vec<f32>,
    pub iter_num: u64,
    pub best_val_loss: f64,
    pub itos: Vec<String>,
}

fn w_u32(f: &mut Vec<u8>, v: u32) {
    f.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(f: &mut Vec<u8>, v: u64) {
    f.extend_from_slice(&v.to_le_bytes());
}
fn w_f32(f: &mut Vec<u8>, v: f32) {
    f.extend_from_slice(&v.to_le_bytes());
}
fn w_f64(f: &mut Vec<u8>, v: f64) {
    f.extend_from_slice(&v.to_le_bytes());
}
fn w_str(f: &mut Vec<u8>, s: &str) {
    w_u32(f, s.len() as u32);
    f.extend_from_slice(s.as_bytes());
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.i + n > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated checkpoint"));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> io::Result<String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

pub fn save_checkpoint(
    path: &Path, model: &Gpt, iter_num: u64, best_val_loss: f64, itos: &[String],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let mut f = Vec::new();
    f.extend_from_slice(MAGIC);
    w_u32(&mut f, VERSION);
    w_u32(&mut f, cfg.block_size as u32);
    w_u32(&mut f, cfg.vocab_size as u32);
    w_u32(&mut f, cfg.n_layer as u32);
    w_u32(&mut f, cfg.n_head as u32);
    w_u32(&mut f, cfg.n_embd as u32);
    w_f32(&mut f, cfg.dropout);
    f.push(cfg.bias as u8);
    w_u64(&mut f, iter_num);
    w_f64(&mut f, best_val_loss);
    w_u32(&mut f, itos.len() as u32);
    for c in itos {
        w_str(&mut f, c);
    }
    w_u64(&mut f, model.params.len() as u64);
    for &p in model.params.iter() {
        w_f32(&mut f, p);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::File::create(path)?.write_all(&f)
}

pub fn load_checkpoint(path: &Path) -> io::Result<Checkpoint> {
    let bytes = fs::read(path)?;
    let mut r = Reader { b: &bytes, i: 0 };
    let magic = r.take(8)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad checkpoint magic"));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version {}", version),
        ));
    }
    let cfg = GptConfig {
        block_size: r.u32()? as usize,
        vocab_size: r.u32()? as usize,
        n_layer: r.u32()? as usize,
        n_head: r.u32()? as usize,
        n_embd: r.u32()? as usize,
        dropout: r.f32()?,
        bias: r.take(1)?[0] != 0,
    };
    let iter_num = r.u64()?;
    let best_val_loss = r.f64()?;
    let n_chars = r.u32()? as usize;
    let mut itos = Vec::with_capacity(n_chars);
    for _ in 0..n_chars {
        itos.push(r.string()?);
    }
    let n_params = r.u64()? as usize;
    let raw = r.take(n_params * 4)?;
    let params: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Checkpoint { cfg, params, iter_num, best_val_loss, itos })
}

/// Rebuild a model from a checkpoint (weights copied verbatim, so weight
/// tying etc. are automatically consistent).
pub fn model_from_checkpoint(ckpt: &Checkpoint) -> io::Result<Gpt> {
    let mut rng = crate::rng::Rng::new(0); // init values overwritten below
    let mut model = Gpt::new(ckpt.cfg.clone(), &mut rng);
    if model.params.len() != ckpt.params.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checkpoint param count mismatch: {} vs {}",
                ckpt.params.len(),
                model.params.len()
            ),
        ));
    }
    model.params.copy_from_slice(&ckpt.params);
    Ok(model)
}
