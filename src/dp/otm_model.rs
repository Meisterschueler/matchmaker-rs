//! Port of Parangonar 3.4.1's `AlignmentTransformer` and `tokenize`, on CPU.
//! Upstream: `parangonar/match/pretrained_models.py`, `parangonar/match/online_matchers.py`.
//!
//! ```
//! use matchmaker_rs::dp::otm_model::{tokenize, OtmModel};
//! let model = OtmModel::bundled().unwrap();
//! let tokens = tokenize(&[vec![60, 64, 67]], &[60]).unwrap();
//! assert_eq!(model.forward(&tokens).unwrap().len(), 26 * 2);
//! ```

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const N: usize = 26;
const D: usize = 64;
const SOURCE: &str = "9ec8a1b8e6731c01a9bb359635dab3fa0b9e60045bbbf18f27227237b249af97";

fn invalid(key: &str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        key: key.to_owned(),
        reason: reason.into(),
    }
}

/// Encode already ordered score pitch sets and performance pitches as OTM tokens.
///
/// Invalid score pitches consume their original slots but remain zero, exactly as upstream.
/// The caller supplies score sets in reference iteration order.
pub fn tokenize(score: &[Vec<i32>], performance: &[i32]) -> Result<[[usize; 7]; 26]> {
    if score.len() > 16 || performance.len() > 8 {
        return Err(invalid(
            "otm_model",
            "token window allows at most 16 score onsets and 8 performance notes",
        ));
    }
    let mut result = [[90; 7]; N];
    for (i, &pitch) in performance.iter().enumerate() {
        if !(20..=110).contains(&pitch) {
            return Err(invalid(
                "pitch",
                format!("{pitch} maps outside the OTM vocabulary"),
            ));
        }
        result[i] = [(pitch - 20) as usize; 7];
    }
    result[performance.len()] = [89; 7];
    for (i, pitches) in score.iter().enumerate() {
        let token = &mut result[performance.len() + 1 + i];
        *token = [0; 7];
        for (slot, &pitch) in pitches.iter().take(7).enumerate() {
            if (21..=108).contains(&pitch) {
                token[slot] = (pitch - 20) as usize;
            }
        }
    }
    Ok(result)
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    source_sha256: String,
    sha256: String,
    tensors: Vec<TensorInfo>,
}

#[derive(Deserialize)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    offset: usize,
}

struct Layer {
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    attn_w: Vec<f32>,
    attn_b: Vec<f32>,
    ff1_w: Vec<f32>,
    ff1_b: Vec<f32>,
    ff2_w: Vec<f32>,
    ff2_b: Vec<f32>,
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
}

/// Immutable, validated OTM model.
pub struct OtmModel {
    position: Vec<f32>,
    embedding: Vec<f32>,
    layers: Vec<Layer>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl OtmModel {
    /// Load the model packaged with the library, checking its manifest and checksum.
    pub fn bundled() -> Result<Self> {
        Self::from_bytes(
            include_bytes!("../../assets/otm/model.bin"),
            include_str!("../../assets/otm/model.json"),
        )
    }

    /// Validate and load a compact little-endian float32 artifact and its JSON manifest.
    pub fn from_bytes(weights: &[u8], manifest: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_str(manifest)
            .map_err(|e| invalid("otm_model", format!("invalid manifest: {e}")))?;
        if manifest.format != "matchmaker-otm"
            || manifest.version != 1
            || manifest.source_sha256 != SOURCE
        {
            return Err(invalid(
                "otm_model",
                "unsupported format, version or source",
            ));
        }
        let digest = format!("{:x}", Sha256::digest(weights));
        if digest != manifest.sha256 {
            return Err(invalid("otm_model", "weight checksum mismatch"));
        }
        let mut index = 0;
        let mut offset = 0usize;
        let mut tensor = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let info = manifest
                .tensors
                .get(index)
                .ok_or_else(|| invalid("otm_model", format!("missing tensor {name}")))?;
            if info.name != name || info.shape != shape || info.offset != offset {
                return Err(invalid(
                    "otm_model",
                    format!("invalid shape or offset for {name}"),
                ));
            }
            let count = shape.iter().product::<usize>();
            let end = offset + count * 4;
            let bytes = weights
                .get(offset..end)
                .ok_or_else(|| invalid("otm_model", format!("truncated tensor {name}")))?;
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("otm_model", format!("non-finite tensor {name}")));
            }
            offset = end;
            index += 1;
            Ok(values)
        };
        let position = tensor("positional_encoder.pos_encoding", &[N, 1, D])?;
        let embedding = tensor("embedding.weight", &[91, D])?;
        let mut layers = Vec::with_capacity(6);
        for i in 0..6 {
            let prefix = format!("transformerDECODER.layers.{i}.");
            let mut get =
                |suffix: &str, shape: &[usize]| tensor(&format!("{prefix}{suffix}"), shape);
            layers.push(Layer {
                qkv_w: get("self_attn.in_proj_weight", &[3 * D, D])?,
                qkv_b: get("self_attn.in_proj_bias", &[3 * D])?,
                attn_w: get("self_attn.out_proj.weight", &[D, D])?,
                attn_b: get("self_attn.out_proj.bias", &[D])?,
                ff1_w: get("linear1.weight", &[D, D])?,
                ff1_b: get("linear1.bias", &[D])?,
                ff2_w: get("linear2.weight", &[D, D])?,
                ff2_b: get("linear2.bias", &[D])?,
                norm1_w: get("norm1.weight", &[D])?,
                norm1_b: get("norm1.bias", &[D])?,
                norm2_w: get("norm2.weight", &[D])?,
                norm2_b: get("norm2.bias", &[D])?,
            });
        }
        let out_w = tensor("out.weight", &[2, D])?;
        let out_b = tensor("out.bias", &[2])?;
        if index != manifest.tensors.len() || offset != weights.len() {
            return Err(invalid("otm_model", "unexpected extra tensor or bytes"));
        }
        Ok(Self {
            position,
            embedding,
            layers,
            out_w,
            out_b,
        })
    }

    /// Return row-major 26 × 2 logits for one token window.
    pub fn forward(&self, tokens: &[[usize; 7]; 26]) -> Result<Vec<f32>> {
        self.run(tokens, |_, _| {})
    }

    fn run(
        &self,
        tokens: &[[usize; 7]; N],
        mut observe: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>> {
        let mut x = vec![0.0; N * D];
        for (p, token) in tokens.iter().enumerate() {
            for &id in token {
                if id >= 91 {
                    return Err(invalid(
                        "otm_model",
                        format!("token {id} outside vocabulary"),
                    ));
                }
                for d in 0..D {
                    x[p * D + d] += self.embedding[id * D + d];
                }
            }
            for d in 0..D {
                x[p * D + d] += self.position[p * D + d];
            }
        }
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            observe(i, &x);
        }
        Ok(linear(&x, &self.out_w, &self.out_b, 2))
    }
}

fn linear(input: &[f32], weight: &[f32], bias: &[f32], output_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; N * output_dim];
    for p in 0..N {
        for o in 0..output_dim {
            let mut sum = bias[o];
            for d in 0..D {
                sum += input[p * D + d] * weight[o * D + d];
            }
            output[p * output_dim + o] = sum;
        }
    }
    output
}

fn norm_residual(x: &[f32], residual: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; N * D];
    for p in 0..N {
        let row = &mut out[p * D..(p + 1) * D];
        for d in 0..D {
            row[d] = x[p * D + d] + residual[p * D + d];
        }
        let mean = row.iter().sum::<f32>() / D as f32;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
        let scale = 1.0 / libm::sqrtf(variance + 1e-5);
        for d in 0..D {
            row[d] = (row[d] - mean) * scale * weight[d] + bias[d];
        }
    }
    out
}

impl Layer {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let qkv = linear(x, &self.qkv_w, &self.qkv_b, 3 * D);
        let mut attention = vec![0.0; N * D];
        for h in 0..8 {
            for p in 0..N {
                let mut scores = [0.0f32; N];
                for j in 0..N {
                    let mut dot = 0.0;
                    for d in 0..8 {
                        dot += qkv[p * 3 * D + h * 8 + d] * qkv[j * 3 * D + D + h * 8 + d];
                    }
                    scores[j] = dot * (1.0 / libm::sqrtf(8.0));
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for v in &mut scores {
                    *v = libm::expf(*v - max);
                }
                let sum: f32 = scores.iter().sum();
                for j in 0..N {
                    for d in 0..8 {
                        attention[p * D + h * 8 + d] +=
                            (scores[j] / sum) * qkv[j * 3 * D + 2 * D + h * 8 + d];
                    }
                }
            }
        }
        let projected = linear(&attention, &self.attn_w, &self.attn_b, D);
        let x = norm_residual(x, &projected, &self.norm1_w, &self.norm1_b);
        let mut ff = linear(&x, &self.ff1_w, &self.ff1_b, D);
        for v in &mut ff {
            *v = v.max(0.0);
        }
        let ff = linear(&ff, &self.ff2_w, &self.ff2_b, D);
        norm_residual(&x, &ff, &self.norm2_w, &self.norm2_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_layers_match_pytorch() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/golden/otm_model_layers/fixture.json"
        ))
        .unwrap();
        let mut tokens = [[0; 7]; N];
        for (row, input) in tokens.iter_mut().zip(fixture["tokens"].as_array().unwrap()) {
            for (value, input) in row.iter_mut().zip(input.as_array().unwrap()) {
                *value = input.as_u64().unwrap() as usize;
            }
        }
        OtmModel::bundled()
            .unwrap()
            .run(&tokens, |layer, actual| {
                for (i, (&a, e)) in actual
                    .iter()
                    .zip(fixture["layers"][layer].as_array().unwrap())
                    .enumerate()
                {
                    let e = e.as_f64().unwrap() as f32;
                    assert!(
                        (a - e).abs() < 3e-4,
                        "layer {layer}, element {i}: {a} != {e}"
                    );
                }
            })
            .unwrap();
    }
}
