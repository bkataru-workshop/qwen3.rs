use memmapix::Mmap;
use std::any::Any;
use std::fs::File;
/// Inference for GGUF Qwen-3 models in pure Rust
use std::io::{self, BufRead};
use std::path::Path;
use std::str::RSplitTerminator;

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

#[derive(Debug)]
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Box<[f32]>, // (vocab_size, dim)
    // weights for rmsnorms in each layer
    rms_att_weight: Box<[f32]>, // (layer, dim)
    rms_ffn_weight: Box<[f32]>, // (layer, dim)
    // weights for matmuls
    wq: Box<[f32]>,      // (layer, dim, n_heads * head_dim)
    wk: Box<[f32]>,      // (layer, dim, n_kv_heads * head_dim)
    wv: Box<[f32]>,      // (layer, dim, m_kv_heads * head_dim)
    wo: Box<[f32]>,      // (layer, n_heads * head_dim, dim)
    wq_norm: Box<[f32]>, // (layer, head_dim)
    wk_norm: Box<[f32]>, // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    w1: Box<[f32]>, // (layer, dim, hidden_dim)
    w2: Box<[f32]>, // (layer, hidden_dim, dim)
    w3: Box<[f32]>, // (layer, dim, hidden_dim)
    // final rmsnorm
    rms_final_weight: Box<[f32]>, // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    wcls: Box<[f32]>,
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
    config: Config,              // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState,             // buffers for the "wave" of activations in the forward pass
    fd: File,                    // file handler for memory mapping
    _mmap: Mmap,                 // keep mmap alive; dropping it unmaps the file
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

impl TransformerWeights {
    /// Memory map weights from a byte slice at a given offset
    pub fn mmap(
        data: &[u8],
        config: &Config,
        header_offset: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Reinterpret the byte slice as f32 slice
        let float_data = Self::bytes_as_floats(&data[header_offset..])?;
        let mut offset = 0;

        macro_rules! consume {
            ($name:expr, $len:expr) => {{
                let slice = &float_data[offset..offset + $len];
                offset += $len;
                slice.to_vec().into_boxed_slice()
            }};
        }

        Ok(Self {
            wcls: consume!("wcls", config.vocab_size * config.dim),
            rms_final_weight: consume!("rms_final_weight", config.dim),
            token_embedding_table: consume!(
                "token_embedding_table",
                config.vocab_size * config.dim
            ),
            wk: consume!("wk", config.dim * config.n_kv_heads * config.head_dim),
            wk_norm: consume!("wk_norm", config.head_dim),
            rms_att_weight: consume!("rms_att_weight", config.dim),
            wo: consume!("wo", config.n_heads * config.head_dim * config.dim),
            wq: consume!("wq", config.dim * config.n_heads * config.head_dim),
            wq_norm: consume!("wq_norm", config.head_dim),
            wv: consume!("wv", config.dim * config.n_kv_heads * config.head_dim),
            w2: consume!("w2", config.hidden_dim * config.dim),
            w3: consume!("w3", config.dim * config.hidden_dim),
            rms_ffn_weight: consume!("rms_ffn_weight", config.dim),
            w1: consume!("w1", config.dim * config.hidden_dim),
        })
    }

    fn bytes_as_floats(data: &[u8]) -> Result<&[f32], Box<dyn std::error::Error>> {
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

        let weights = TransformerWeights::mmap(&mmap, config, header_offset)?;
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

mod how_to_memory_map {
    use std::fs::File;
    use std::io::Read;

    use memmapix::Mmap;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("./main.rs")?;

        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        let mmap = unsafe { Mmap::map(&file)? };

        let foo = &mmap[..];

        assert_eq!(&contents[..], &mmap[..]);

        Ok(())
    }
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
