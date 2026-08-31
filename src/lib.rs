use memmapix::Mmap;
use rayon::prelude::*;
use std::fs::File;
/// Inference for GGUF Qwen-3 models in pure Rust
use std::io::{self, BufRead};
use std::path::Path;

// ----------------------------------------------------------------------------
// Transformer model
#[derive(Debug, Copy, Clone)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size
    seq_len: usize,    // max sequence length
    head_dim: usize,   // attention dimension
}

#[derive(Debug, Clone, Copy)]
struct TransformerWeights<'a> {
    // token embedding table
    token_embedding_table: &'a [f32], // (vocab_size, dim)
    // weights for rmsnorms in each layer
    rms_att_weight: &'a [f32], // (layer, dim)
    rms_ffn_weight: &'a [f32], // (layer, dim)
    // weights for matmuls
    wq: &'a [f32],      // (layer, dim, n_heads * head_dim)
    wk: &'a [f32],      // (layer, dim, n_kv_heads * head_dim)
    wv: &'a [f32],      // (layer, dim, n_kv_heads * head_dim)
    wo: &'a [f32],      // (layer, n_heads * head_dim, dim)
    wq_norm: &'a [f32], // (layer, head_dim)
    wk_norm: &'a [f32], // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    w1: &'a [f32], // (layer, dim, hidden_dim)
    w2: &'a [f32], // (layer, hidden_dim, dim)
    w3: &'a [f32], // (layer, dim, hidden_dim)
    // final rmsnorm
    rms_final_weight: &'a [f32], // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    wcls: &'a [f32],
}

#[derive(Debug)]
struct RunState {
    // current wave of activations
    x: Box<[f32]>,      // activation at current time stamp (dim,)
    xb: Box<[f32]>,     // buffer (dim,)
    xb2: Box<[f32]>,    // an additional buffer just for convenience (dim,)
    xb3: Box<[f32]>,    // an additional buffer just for convenience (att_head_dim,)
    hb: Box<[f32]>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Box<[f32]>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Box<[f32]>,      // query (att_head_dim,)
    k: Box<[f32]>,      // key (dim,)
    v: Box<[f32]>,      // value (dim,)
    att: Box<[f32]>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Box<[f32]>, // output logits
    // kv cache
    key_cache: Box<[f32]>,   // (layer, seq_len, dim)
    value_cache: Box<[f32]>, // (layer, seq_len, dim)
}

#[derive(Debug)]
struct Transformer {
    config: Config, // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights<'static>, // the weights of the model, unsafe lifetime extension tied to owned _mmap
    state: RunState, // buffers for the "wave" of activations in the forward pass
    fd: File,        // file handler for memory mapping
    _mmap: Mmap,     // keep mmap alive; dropping it unmaps the file
    // data: Box<[f32]>,            // memory mapped data pointer
    file_size: u64, // size of the checkpoint file in bytes
}

impl RunState {
    pub fn calloc(p: Config) -> Self {
        let att_head_dim = p.n_heads * p.head_dim;
        let kv_dim = p.n_kv_heads * p.head_dim; // 1024

        Self {
            x: vec![0.0; p.dim].into_boxed_slice(),
            xb: vec![0.0; p.dim].into_boxed_slice(),
            xb2: vec![0.0; p.dim].into_boxed_slice(),
            xb3: vec![0.0; att_head_dim].into_boxed_slice(),
            hb: vec![0.0; p.hidden_dim].into_boxed_slice(),
            hb2: vec![0.0; p.hidden_dim].into_boxed_slice(),
            q: vec![0.0; att_head_dim].into_boxed_slice(),
            k: vec![0.0; kv_dim].into_boxed_slice(),
            v: vec![0.0; kv_dim].into_boxed_slice(),
            att: vec![0.0; p.n_heads * p.seq_len].into_boxed_slice(),
            logits: vec![0.0; p.vocab_size].into_boxed_slice(),
            key_cache: vec![0.0; p.n_layers * p.seq_len * kv_dim].into_boxed_slice(),
            value_cache: vec![0.0; p.n_layers * p.seq_len * kv_dim].into_boxed_slice(),
        }
    }
}

impl<'a> TransformerWeights<'a> {
    /// Memory map weights from a byte slice at a given offset
    pub fn mmap(
        data: &'a [u8],
        config: &Config,
        header_offset: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Reinterpret the byte slice as f32 slice
        let float_data = Self::bytes_as_floats(&data[header_offset..])?;
        let mut offset = 0;

        let mut consume = |len: usize| -> Result<&'a [f32], Box<dyn std::error::Error>> {
            if offset + len > float_data.len() {
                return Err("Attempted to slice past mapped binary boundary".into());
            }
            let slice = &float_data[offset..offset + len];
            offset += len;
            Ok(slice)
        };

        Ok(Self {
            wcls: consume(config.vocab_size * config.dim)?,
            rms_final_weight: consume(config.dim)?,
            token_embedding_table: consume(config.vocab_size * config.dim)?,
            wk: consume(config.n_layers * config.dim * config.n_kv_heads * config.head_dim)?,
            wk_norm: consume(config.n_layers * config.head_dim)?,
            rms_att_weight: consume(config.n_layers * config.dim)?,
            wo: consume(config.n_layers * config.n_heads * config.head_dim * config.dim)?,
            wq: consume(config.n_layers * config.dim * config.n_heads * config.head_dim)?,
            wq_norm: consume(config.n_layers * config.head_dim)?,
            wv: consume(config.n_layers * config.dim * config.n_kv_heads * config.head_dim)?,
            w2: consume(config.n_layers * config.hidden_dim * config.dim)?,
            w3: consume(config.n_layers * config.dim * config.hidden_dim)?,
            rms_ffn_weight: consume(config.n_layers * config.dim)?,
            w1: consume(config.n_layers * config.dim * config.hidden_dim)?,
        })
    }

    fn bytes_as_floats(data: &'a [u8]) -> Result<&'a [f32], Box<dyn std::error::Error>> {
        if data.len() % 4 != 0 {
            return Err("Byte slice length must be a multiple of 4".into());
        }
        if data.as_ptr() as usize % 4 != 0 {
            return Err("Data is not 4-byte aligned".into());
        }

        unsafe {
            let ptr = data.as_ptr() as *const f32;
            let len = data.len() / 4;
            Ok(std::slice::from_raw_parts(ptr, len))
        }
    }
}

impl Transformer {
    // read GGUF
    pub fn read_checkpoint(
        checkpoint_path: &str,
        config: &mut Config,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(checkpoint_path)?;
        let file_size = file.metadata()?.len();

        // Memory map the file
        let mmap = unsafe { Mmap::map(&file)? };

        // Skip GGUF header (hardcoded for now, but parse it properly later)
        let header_offset = 5951648;

        // Erase lifetime safely: `mmap` is stored permanently in `Transformer`
        // and will outlive `weights`.
        let static_bytes: &'static [u8] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

        let weights = TransformerWeights::mmap(static_bytes, config, header_offset)?;
        let state = RunState::calloc(*config);

        Ok(Self {
            config: *config,
            weights,
            state,
            fd: file,
            _mmap: mmap, // Keep the mmap alive
            file_size,
        })
    }

    pub fn build(checkpoint_path: &str, config: &mut Config) -> Self {
        match Self::read_checkpoint(checkpoint_path, config) {
            Ok(transformer) => transformer,
            Err(error) => {
                eprintln!("Error building Transformer: {}", error);
                std::process::exit(1);
            }
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Box<[f32]> {
        let config = &self.config;
        let weights = &self.weights;
        let state = &mut self.state;

        let kv_dim = config.n_kv_heads * config.head_dim;
        let kv_mul = config.n_heads / config.n_kv_heads;
        let att_head_dim = config.n_heads * config.head_dim;

        let layer_offset = 62923776 / 4; // offset to the GGUF next layer for the same tensor type TODO

        let start = token * config.dim;
        let end = start + config.dim;

        // copy the token embedding into s->x, STARTING POINT - x is passing through.
        state.x[..config.dim].copy_from_slice(&weights.token_embedding_table[start..end]);

        // forward all the layers
        for l in 0..config.n_layers {
            // kv cache
            let loff = l * config.seq_len * kv_dim;
            let cache_idx = loff + pos * kv_dim;
            let w_off = l * layer_offset;

            // attention rmsnorm
            state.xb.copy_from_slice(&state.x);
            rmsnorm(&mut state.xb, &weights.rms_att_weight[w_off..], config.dim);

            // query projection
            matmul(
                &mut state.q,
                &state.xb,
                &weights.wq[w_off..],
                config.dim,
                att_head_dim,
            );

            // mutable sub-slices of the KV cache for this specific layer/token
            let k_slice = &mut state.key_cache[cache_idx..cache_idx + kv_dim];
            let v_slice = &mut state.value_cache[cache_idx..cache_idx + kv_dim];

            // key/value projections
            matmul(k_slice, &state.xb, &weights.wk[w_off..], config.dim, kv_dim);
            matmul(v_slice, &state.xb, &weights.wv[w_off..], config.dim, kv_dim);

            // RoPE relative positional encoding
            for h in 0..config.n_heads {
                // query head
                let q_start = h * config.head_dim;
                let q_head = &mut state.q[q_start..q_start + config.head_dim];

                // key head (conditionally)
                let mut k_head = if h < config.n_kv_heads {
                    let k_start = h * config.head_dim;
                    Some(&mut k_slice[k_start..k_start + config.head_dim])
                } else {
                    None
                };

                // apply RMSNorm to query head
                rmsnorm(q_head, &weights.wq_norm[w_off..], config.head_dim);

                // apply RMSNorm to key head if within n_kv_heads
                if let Some(ref mut k) = k_head {
                    rmsnorm(k, &weights.wk_norm[w_off..], config.head_dim);
                }

                // apply rotary position encoding
                for i in 0..(config.head_dim / 2) {
                    let freq = 1.0 / 1000000.0_f32.powf(i as f32 / (config.head_dim as f32 / 2.0));
                    let fcr = (pos as f32 * freq).cos();
                    let fci = (pos as f32 * freq).sin();

                    // rotate query head
                    let x_q = q_head[i];
                    let y_q = q_head[i + config.head_dim / 2];
                    q_head[i] = x_q * fcr - y_q * fci;
                    q_head[i + config.head_dim / 2] = x_q * fci + y_q * fcr;

                    // rotate key head if within n_kv_heads
                    if let Some(ref mut k) = k_head {
                        let x_k = k[i];
                        let y_k = k[i + config.head_dim / 2];
                        k[i] = x_k * fcr - y_k * fci;
                        k[i + config.head_dim / 2] = x_k * fci + y_k * fcr;
                    }
                }
            }

            // multihead attention. iterate over all heads
            // parallelized across query heads
            let head_dim = config.head_dim;
            let seq_len = config.seq_len;
            let head_dim_sqrt = (head_dim as f32).sqrt();

            // Rebind references so Rayon closures only capture the needed slices
            let key_cache = &state.key_cache;
            let value_cache = &state.value_cache;

            state
                .q
                .par_chunks(head_dim)
                .zip(state.att.par_chunks_mut(seq_len))
                .zip(state.xb3.par_chunks_mut(head_dim))
                .enumerate()
                .for_each(|(h, ((q_head, att_head), xb3_head))| {
                    let kv_head = h / kv_mul;
                    let kv_head_offset = kv_head * head_dim;

                    // calculate attention scores: dot product of Q and K for timesteps 0..=pos
                    for t in 0..=pos {
                        let k_offset = loff + t * kv_dim + kv_head_offset;
                        let k_head = &key_cache[k_offset..k_offset + head_dim];

                        let score: f32 = q_head.iter().zip(k_head).map(|(q, k)| q * k).sum();
                        att_head[t] = score / head_dim_sqrt;
                    }

                    // softmax the scores to get attention weights, from 0..=pos
                    softmax(att_head, pos + 1);

                    // weighted sum of the values, store back into xb3
                    xb3_head.fill(0.0);
                    for t in 0..=pos {
                        let v_offset = loff + t * kv_dim + kv_head_offset;
                        // get the value vector for this head and at this timestep
                        let v_head = &value_cache[v_offset..v_offset + head_dim];
                        // get the attention weight for this timestep
                        let a = att_head[t];
                        // accumulate the weighted value into xb3
                        for i in 0..head_dim {
                            xb3_head[i] += a * v_head[i];
                        }
                    }
                });

            // output projection
            matmul(
                &mut state.xb2,
                &state.xb3,
                &weights.wo[w_off..],
                att_head_dim,
                config.dim,
            );

            // residual connection back into state.x
            for i in 0..config.dim {
                state.x[i] += state.xb2[i];
            }

            // ffn rmsnorm
            state.xb.copy_from_slice(&state.x);
            rmsnorm(&mut state.xb, &weights.rms_ffn_weight[w_off..], config.dim);

            matmul(
                &mut state.hb,
                &state.xb,
                &weights.w1[w_off..],
                config.dim,
                config.hidden_dim,
            );
            matmul(
                &mut state.hb2,
                &state.xb,
                &weights.w3[w_off..],
                config.dim,
                config.hidden_dim,
            );

            // SwiGLU non-linearity
            for i in 0..config.hidden_dim {
                let val = state.hb2[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                let silu = val * (1.0 / (1.0 + (-val).exp()));
                state.hb2[i] = silu * state.hb[i]; // elementwise multiply with w3(x)
            }

            // matmul to get the final ffn output
            // quantize(&state.hq, state.hb, config.hidden_dim);
            matmul(
                &mut state.xb,
                &state.hb2,
                &weights.w2[w_off..],
                config.hidden_dim,
                config.dim,
            );

            // residual connection
            for i in 0..config.dim {
                state.x[i] += state.xb[i];
            }
        }

        // rmsnorm right before logiting
        rmsnorm(&mut state.x, &weights.rms_final_weight, config.dim);

        matmul(
            &mut state.logits,
            &state.x,
            &weights.wcls,
            config.dim,
            config.vocab_size,
        );

        state.logits.clone()
    }
}

impl Config {
    fn read_lines<P>(filename: P) -> io::Lines<io::BufReader<File>>
    where
        P: AsRef<Path> + std::fmt::Display,
    {
        let file = match File::open(&filename) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("Failed to open {} : {}", &filename, error);
                std::process::exit(1);
            }
        };
        io::BufReader::new(file).lines()
    }

    // load the GGUF config file
    pub fn load(filename: Option<String>) -> Self {
        let filename = filename.unwrap_or_else(|| "header.txt".to_string());

        let lines = Self::read_lines(filename);
        let mut config: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

        let format_parsing_error_message = |key: &str, value: &str| {
            format!("Error parsing value '{}' as usize for key {}", value, key)
        };

        for line in lines.map_while(Result::ok) {
            match line.split_once("=") {
                Some((key, value)) => match key {
                    "QWEN3_EMBEDDING_LENGTH" => {
                        config.insert(
                            "dim",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_FEED_FORWARD_LENGTH" => {
                        config.insert(
                            "hidden_dim",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_BLOCK_COUNT" => {
                        config.insert(
                            "n_layers",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_ATTENTION_HEAD_COUNT" => {
                        config.insert(
                            "n_heads",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_ATTENTION_HEAD_COUNT_KV" => {
                        config.insert(
                            "n_kv_heads",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_CONTEXT_LENGTH" => {
                        config.insert(
                            "seq_len",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "QWEN3_ATTENTION_KEY_LENGTH" => {
                        config.insert(
                            "head_dim",
                            value
                                .parse::<usize>()
                                .expect(&format_parsing_error_message(key, value)),
                        );
                    }
                    "TOKENIZER_GGML_TOKENS" => {
                        const ARRAY_LENGTH_KEY: &str = "ARRAY_LENGTH=";

                        if let Some(start) = value.find(ARRAY_LENGTH_KEY) {
                            let start = start + ARRAY_LENGTH_KEY.len();
                            let value = value[start..].to_string();
                            config.insert(
                                "vocab_size",
                                value
                                    .parse::<usize>()
                                    .expect(&format_parsing_error_message(key, &value)),
                            );
                        } else {
                            eprintln!("No key named '{}' found in config", ARRAY_LENGTH_KEY);
                            std::process::exit(1);
                        }
                    }
                    _ => {}
                },
                None => {}
            }
        }

        if config.len() != 8 {
            eprintln!("Invalid or corrupted config, didn't find exactly eight keys");
            std::process::exit(1);
        }

        Self {
            dim: config["dim"],
            hidden_dim: config["hidden_dim"],
            n_layers: config["n_layers"],
            n_heads: config["n_heads"],
            n_kv_heads: config["n_kv_heads"],
            seq_len: config["seq_len"],
            head_dim: config["head_dim"],
            vocab_size: config["vocab_size"],
        }
    }
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

pub fn rmsnorm(x: &mut [f32], weight: &[f32], size: usize) {
    // calculate sum of squares
    // iterator enables auto-vectorization
    let ss = x[..size].iter().map(|&v| v * v).sum::<f32>() / size as f32 + 1e-6;
    let scale = 1.0 / ss.sqrt();

    // normalize and scale
    // in-place for cache efficiency
    for j in 0..size {
        x[j] *= scale * weight[j];
    }
}

pub fn softmax(x: &mut [f32], size: usize) {
    // find max value (for numerical stability)
    let max_val = x[..size]
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .copied()
        .unwrap_or(f32::NAN);

    // exp and sum
    // TODO: what about this? how does an iterator-based approach compare
    // x = x.iter().map(|c| (c - max_val).exp()).collect();
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
    }
    let sum = x[..size].iter().sum::<f32>();

    // normalize
    for i in 0..size {
        x[i] /= sum;
    }
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // TODO:
    // the C version parallelizes using OpenMP via
    // #pragma omp parallel for private(i)
    // figure out how to do this in Rust
    //
    // .par_iter_mut() is Rust's equivalent of OpenMP's parallel for!
    // it automatically divides work across CPU cores
    xout[..d]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            let mut val = 0.0_f32;

            // grab the specific row of weights for this iteration
            // (helps the compiler remove bounds checks inside the hot loop)
            let w_row = &w[i * n..i * n + n];

            for j in 0..n {
                val += w_row[j] * x[j];
            }

            *out_val = val;
        });
}

fn main() {
    println!("hello world");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_load() {
        let config = Config::load(Some("header.txt".to_string()));

        println!("{:?}", config);

        assert!(true);
    }
}
