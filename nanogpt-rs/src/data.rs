//! Data loading for the char-level shakespeare dataset (uint16 .bin files,
//! same format nanoGPT's `prepare.py` produces), plus a Rust-native
//! `prepare` implementation and a minimal JSON reader for `meta.json`.

use crate::rng::Rng;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub struct Dataset {
    pub train: Vec<u16>,
    pub val: Vec<u16>,
    pub vocab_size: usize,
    /// id -> char mapping (empty when unknown)
    pub itos: Vec<String>,
}

impl Dataset {
    /// Load train.bin / val.bin (little-endian uint16 token streams).
    /// vocab_size comes from meta.json when present, otherwise it is
    /// inferred as max(token id) + 1 over both splits.
    pub fn load(data_dir: &str) -> io::Result<Self> {
        let train = load_bin(&Path::new(data_dir).join("train.bin"))?;
        let val = load_bin(&Path::new(data_dir).join("val.bin"))?;
        let meta_path = Path::new(data_dir).join("meta.json");
        // meta.json is our native metadata format; if only Python's meta.pkl
        // exists, convert it once via a small python one-liner.
        if !meta_path.exists() {
            let pkl = Path::new(data_dir).join("meta.pkl");
            if pkl.exists() {
                convert_meta_pkl(&pkl, &meta_path);
            }
        }
        let (vocab_size, itos) = if meta_path.exists() {
            match load_meta(&meta_path) {
                Some(m) => m,
                None => infer_vocab(&train, &val),
            }
        } else {
            infer_vocab(&train, &val)
        };
        Ok(Dataset { train, val, vocab_size, itos })
    }

    /// Sample a random batch exactly like nanoGPT's `get_batch`:
    /// x[i] = data[ix..ix+block_size], y[i] = data[ix+1..ix+1+block_size].
    pub fn get_batch(&self, split: Split, batch_size: usize, block_size: usize,
                      rng: &mut Rng, x: &mut Vec<u32>, y: &mut Vec<u32>) {
        let data = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };
        assert!(data.len() > block_size + 1, "split too small for block_size");
        x.resize(batch_size * block_size, 0);
        y.resize(batch_size * block_size, 0);
        for b in 0..batch_size {
            let ix = rng.rand_below(data.len() - block_size);
            for i in 0..block_size {
                x[b * block_size + i] = data[ix + i] as u32;
                y[b * block_size + i] = data[ix + i + 1] as u32;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Split {
    Train,
    Val,
}

fn load_bin(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "odd byte length in bin file"));
    }
    Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn infer_vocab(train: &[u16], val: &[u16]) -> (usize, Vec<String>) {
    let mx = train.iter().chain(val.iter()).max().copied().unwrap_or(0) as usize;
    (mx + 1, Vec::new())
}

/// Best-effort one-time conversion of Python's meta.pkl into meta.json.
fn convert_meta_pkl(pkl: &Path, json_out: &Path) {
    let script = "import pickle,json,sys;\
m=pickle.load(open(sys.argv[1],'rb'));\
json.dump({'vocab_size':m['vocab_size'],'itos':[m['itos'][i] for i in range(m['vocab_size'])]},\
open(sys.argv[2],'w',encoding='utf-8'))";
    let status = std::process::Command::new("python")
        .args(["-c", &script, &pkl.to_string_lossy(), &json_out.to_string_lossy()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => println!("converted {} -> {}", pkl.display(), json_out.display()),
        _ => eprintln!(
            "note: could not convert meta.pkl to meta.json (python unavailable?); \
             sampling output will show token ids instead of text"
        ),
    }
}

// ---------------------------------------------------------------------------
// Rust-native data preparation (equivalent of data/shakespeare_char/prepare.py)
// ---------------------------------------------------------------------------

/// Read input.txt, build the char vocabulary, write train.bin / val.bin
/// (uint16 LE) and meta.json. Returns (vocab_size, itos).
pub fn prepare(input_path: &Path, out_dir: &Path) -> io::Result<(usize, Vec<String>)> {
    let data = fs::read_to_string(input_path)?;
    println!("length of dataset in characters: {}", data.len());

    let mut chars: Vec<char> = data.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    chars.sort();
    let vocab_size = chars.len();
    println!("all the unique characters: {}", chars.iter().collect::<String>());
    println!("vocab size: {}", vocab_size);

    let mut stoi_map = std::collections::HashMap::<char, usize>::new();
    for (i, &ch) in chars.iter().enumerate() {
        stoi_map.insert(ch, i);
    }

    let n = data.chars().count();
    // Python does int(len(data)*0.9) on characters; replicate exactly:
    let n_train = (n as f64 * 0.9) as usize;

    let mut train_ids: Vec<u16> = Vec::with_capacity(n_train);
    let mut val_ids: Vec<u16> = Vec::with_capacity(n - n_train);
    for (i, ch) in data.chars().enumerate() {
        let id = stoi_map[&ch] as u16;
        if i < n_train {
            train_ids.push(id);
        } else {
            val_ids.push(id);
        }
    }
    println!("train has {} tokens", train_ids.len());
    println!("val has {} tokens", val_ids.len());

    fs::create_dir_all(out_dir)?;
    write_u16_bin(&out_dir.join("train.bin"), &train_ids)?;
    write_u16_bin(&out_dir.join("val.bin"), &val_ids)?;

    // meta.json: {"vocab_size": N, "itos": ["\n", " ", ...]}
    let mut meta = String::from("{\n  \"vocab_size\": ");
    meta.push_str(&vocab_size.to_string());
    meta.push_str(",\n  \"itos\": [");
    for (i, ch) in chars.iter().enumerate() {
        if i > 0 {
            meta.push_str(", ");
        }
        meta.push('"');
        match ch {
            '\n' => meta.push_str("\\n"),
            '\r' => meta.push_str("\\r"),
            '\t' => meta.push_str("\\t"),
            '"' => meta.push_str("\\\""),
            '\\' => meta.push_str("\\\\"),
            c if (*c as u32) < 0x20 => meta.push_str(&format!("\\u{:04x}", *c as u32)),
            c => meta.push(*c),
        }
        meta.push('"');
    }
    meta.push_str("]\n}\n");
    fs::write(out_dir.join("meta.json"), meta)?;
    Ok((vocab_size, chars.iter().map(|c| c.to_string()).collect()))
}

fn write_u16_bin(path: &Path, ids: &[u16]) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    let mut buf = Vec::with_capacity(ids.len() * 2);
    for id in ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    f.write_all(&buf)
}

// ---------------------------------------------------------------------------
// minimal JSON parsing for meta.json
// ---------------------------------------------------------------------------

pub fn load_meta(path: &Path) -> Option<(usize, Vec<String>)> {
    let text = fs::read_to_string(path).ok()?;
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    let obj = p.parse_value()?;
    match obj {
        Value::Obj(fields) => {
            let mut vocab = None;
            let mut itos = Vec::new();
            for (k, v) in fields {
                match (k.as_str(), v) {
                    ("vocab_size", Value::Num(n)) => vocab = Some(n as usize),
                    ("itos", Value::Arr(items)) => {
                        itos = items
                            .into_iter()
                            .map(|it| match it {
                                Value::Str(s) => s,
                                _ => String::new(),
                            })
                            .collect();
                    }
                    _ => {}
                }
            }
            Some((vocab?, itos))
        }
        _ => None,
    }
}

enum Value {
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
    Bool(bool),
    Null,
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.i).copied()
    }

    fn parse_value(&mut self) -> Option<Value> {
        match self.peek()? {
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b'"' => Some(Value::Str(self.parse_string()?)),
            b't' => {
                self.expect_lit("true")?;
                Some(Value::Bool(true))
            }
            b'f' => {
                self.expect_lit("false")?;
                Some(Value::Bool(false))
            }
            b'n' => {
                self.expect_lit("null")?;
                Some(Value::Null)
            }
            _ => self.parse_number(),
        }
    }

    fn expect_lit(&mut self, lit: &str) -> Option<()> {
        self.skip_ws();
        let end = self.i + lit.len();
        if end <= self.s.len() && &self.s[self.i..end] == lit.as_bytes() {
            self.i = end;
            Some(())
        } else {
            None
        }
    }

    fn parse_obj(&mut self) -> Option<Value> {
        self.skip_ws();
        if self.s.get(self.i) != Some(&b'{') {
            return None;
        }
        self.i += 1;
        let mut fields = Vec::new();
        loop {
            match self.peek()? {
                b'}' => {
                    self.i += 1;
                    return Some(Value::Obj(fields));
                }
                b'"' => {
                    let key = self.parse_string()?;
                    if self.peek()? != b':' {
                        return None;
                    }
                    self.i += 1;
                    let val = self.parse_value()?;
                    fields.push((key, val));
                    match self.peek()? {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Some(Value::Obj(fields));
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }

    fn parse_arr(&mut self) -> Option<Value> {
        self.skip_ws();
        if self.s.get(self.i) != Some(&b'[') {
            return None;
        }
        self.i += 1;
        let mut items = Vec::new();
        loop {
            if self.peek()? == b']' {
                self.i += 1;
                return Some(Value::Arr(items));
            }
            items.push(self.parse_value()?);
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(Value::Arr(items));
                }
                _ => return None,
            }
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        self.skip_ws();
        if self.s.get(self.i) != Some(&b'"') {
            return None;
        }
        self.i += 1;
        let mut out = String::new();
        while self.i < self.s.len() {
            let c = self.s[self.i];
            match c {
                b'"' => {
                    self.i += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'u' => {
                            if self.i + 4 > self.s.len() {
                                return None;
                            }
                            let hex = std::str::from_utf8(&self.s[self.i..self.i + 4]).ok()?;
                            let cp = u32::from_str_radix(hex, 16).ok()?;
                            out.push(char::from_u32(cp)?);
                            self.i += 4;
                        }
                        _ => return None,
                    }
                }
                _ => {
                    // consume one UTF-8 char
                    let rest = std::str::from_utf8(&self.s[self.i..]).ok()?;
                    let ch = rest.chars().next()?;
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
        None
    }

    fn parse_number(&mut self) -> Option<Value> {
        self.skip_ws();
        let start = self.i;
        while self.i < self.s.len()
            && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let txt = std::str::from_utf8(&self.s[start..self.i]).ok()?;
        Some(Value::Num(txt.parse().ok()?))
    }
}
