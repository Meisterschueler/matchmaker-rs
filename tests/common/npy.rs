//! Minimal `.npy` reader for golden data (NumPy format 1.0–3.0, little-endian, C order).
//!
//! Supports plain and structured arrays of `|b1`, `<i4`, `<i8`, `<f4`, `<f8`; every value is
//! returned as `f64`, which is exact for all of them.

use std::path::Path;

#[derive(Debug, Clone, Copy)]
enum Dtype {
    Bool,
    I32,
    I64,
    F32,
    F64,
}

impl Dtype {
    fn parse(descr: &str) -> Result<Self, String> {
        match descr {
            "|b1" | "<b1" | "?" | "|?" => Ok(Dtype::Bool),
            "<i4" => Ok(Dtype::I32),
            "<i8" => Ok(Dtype::I64),
            "<f4" => Ok(Dtype::F32),
            "<f8" => Ok(Dtype::F64),
            d => Err(format!("unsupported dtype {d}")),
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::F64 => 8,
        }
    }

    fn read(self, b: &[u8]) -> f64 {
        match self {
            Dtype::Bool => f64::from(b[0]),
            Dtype::I32 => f64::from(i32::from_le_bytes(b[..4].try_into().unwrap())),
            Dtype::I64 => i64::from_le_bytes(b[..8].try_into().unwrap()) as f64,
            Dtype::F32 => f64::from(f32::from_le_bytes(b[..4].try_into().unwrap())),
            Dtype::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
        }
    }
}

/// A loaded `.npy` array.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    /// Record fields `(name, dtype, byte offset)`; empty for plain arrays.
    fields: Vec<(String, Dtype, usize)>,
    dtype: Option<Dtype>,
    item_size: usize,
    data: Vec<u8>,
}

impl NpyArray {
    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
            return Err("not a .npy file".into());
        }
        let (header_len, start) = match bytes[6] {
            1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
            _ => (
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
                12,
            ),
        };
        let header =
            std::str::from_utf8(&bytes[start..start + header_len]).map_err(|e| e.to_string())?;
        if header.contains("'fortran_order': True") {
            return Err("Fortran order is not supported".into());
        }
        let shape_text = between(header, "'shape': (", ")")?;
        let shape = shape_text
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().map_err(|_| format!("bad shape {shape_text}")))
            .collect::<Result<Vec<usize>, _>>()?;
        let (dtype, fields, item_size) = if header.contains("'descr': [") {
            let mut fields = Vec::new();
            let mut offset = 0;
            // [('name', '<f4'), ('other', '<i8')]
            for field in between(header, "'descr': [", "]")?.split("),") {
                let mut items = field
                    .split(',')
                    .map(|s| s.trim().trim_matches(|c| c == '(' || c == ')' || c == '\''));
                let name = items.next().ok_or("bad field")?.to_string();
                let dtype = Dtype::parse(items.next().ok_or("bad field")?)?;
                fields.push((name, dtype, offset));
                offset += dtype.size();
            }
            (None, fields, offset)
        } else {
            let dtype = Dtype::parse(between(header, "'descr': '", "'")?)?;
            (Some(dtype), Vec::new(), dtype.size())
        };
        let data = bytes[start + header_len..].to_vec();
        if data.len() != shape.iter().product::<usize>() * item_size {
            return Err(format!(
                "data length {} does not match shape {shape:?}",
                data.len()
            ));
        }
        Ok(NpyArray {
            shape,
            fields,
            dtype,
            item_size,
            data,
        })
    }

    /// Number of elements (records).
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// All elements of a plain array.
    pub fn values(&self) -> Vec<f64> {
        let dtype = self.dtype.expect("plain array");
        self.data
            .chunks_exact(self.item_size)
            .map(|b| dtype.read(b))
            .collect()
    }

    /// One field of a structured array.
    pub fn field(&self, name: &str) -> Vec<f64> {
        let (_, dtype, offset) = self
            .fields
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("no field {name}"));
        self.data
            .chunks_exact(self.item_size)
            .map(|rec| dtype.read(&rec[*offset..]))
            .collect()
    }
}

fn between<'a>(s: &'a str, open: &str, close: &str) -> Result<&'a str, String> {
    let start = s.find(open).ok_or_else(|| format!("missing {open}"))? + open.len();
    let end = s[start..]
        .find(close)
        .ok_or_else(|| format!("missing {close}"))?;
    Ok(&s[start..start + end])
}
