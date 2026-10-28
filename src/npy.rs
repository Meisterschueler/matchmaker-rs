//! Minimal reader/writer for NumPy `.npy` files (format 1.0/2.0, little-endian, C order).
//!
//! Used for golden data and for the Python-readable export (constitution IV). Supports
//! plain arrays of `bool`, `i4`, `i8`, `f4`, `f8` and structured arrays whose fields are of
//! those types. Written files load with `numpy.load`.
//!
//! ```
//! use matchmaker_rs::npy::{NpyArray, Column};
//! let bytes = matchmaker_rs::npy::to_bytes(&[2, 2], &Column::F64(vec![1.0, 2.0, 3.0, 4.0]));
//! let array = NpyArray::from_bytes(&bytes).unwrap();
//! assert_eq!(array.shape, vec![2, 2]);
//! assert_eq!(array.to_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
//! ```

use std::path::Path;

use crate::error::{Error, Result};

const MAGIC: &[u8] = b"\x93NUMPY";

/// A column of values with its element type.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// `|b1`
    Bool(Vec<bool>),
    /// `<i4`
    I32(Vec<i32>),
    /// `<i8`
    I64(Vec<i64>),
    /// `<f4`
    F32(Vec<f32>),
    /// `<f8`
    F64(Vec<f64>),
}

impl Column {
    fn descr(&self) -> &'static str {
        match self {
            Column::Bool(_) => "|b1",
            Column::I32(_) => "<i4",
            Column::I64(_) => "<i8",
            Column::F32(_) => "<f4",
            Column::F64(_) => "<f8",
        }
    }

    fn len(&self) -> usize {
        match self {
            Column::Bool(v) => v.len(),
            Column::I32(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::F64(v) => v.len(),
        }
    }

    fn write_element(&self, i: usize, out: &mut Vec<u8>) {
        match self {
            Column::Bool(v) => out.push(u8::from(v[i])),
            Column::I32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
            Column::I64(v) => out.extend_from_slice(&v[i].to_le_bytes()),
            Column::F32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
            Column::F64(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        }
    }
}

/// Element type of a `.npy` array or field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `|b1`
    Bool,
    /// `<i4`
    I32,
    /// `<i8`
    I64,
    /// `<f4`
    F32,
    /// `<f8`
    F64,
}

impl Dtype {
    fn parse(descr: &str) -> Option<Self> {
        match descr {
            "|b1" | "<b1" | "?" | "|?" => Some(Dtype::Bool),
            "<i4" => Some(Dtype::I32),
            "<i8" => Some(Dtype::I64),
            "<f4" => Some(Dtype::F32),
            "<f8" => Some(Dtype::F64),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::F64 => 8,
        }
    }

    fn read_f64(self, b: &[u8]) -> f64 {
        match self {
            Dtype::Bool => f64::from(b[0]),
            Dtype::I32 => f64::from(i32::from_le_bytes(b[..4].try_into().unwrap_or_default())),
            Dtype::I64 => i64::from_le_bytes(b[..8].try_into().unwrap_or_default()) as f64,
            Dtype::F32 => f64::from(f32::from_le_bytes(b[..4].try_into().unwrap_or_default())),
            Dtype::F64 => f64::from_le_bytes(b[..8].try_into().unwrap_or_default()),
        }
    }
}

/// A loaded `.npy` array: header information plus raw little-endian bytes.
#[derive(Debug, Clone)]
pub struct NpyArray {
    /// Array shape.
    pub shape: Vec<usize>,
    /// `None` for plain arrays, otherwise the record fields with their types and byte offsets.
    pub fields: Option<Vec<(String, Dtype, usize)>>,
    /// Element type for plain arrays.
    pub dtype: Option<Dtype>,
    item_size: usize,
    data: Vec<u8>,
}

impl NpyArray {
    /// Read a `.npy` file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        Self::from_bytes(&bytes).map_err(|reason| Error::Npy {
            path: path.to_path_buf(),
            reason,
        })
    }

    /// Parse `.npy` bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::result::Result<Self, String> {
        if bytes.len() < 10 || &bytes[..6] != MAGIC {
            return Err("not a .npy file".into());
        }
        let (header_len, start) = match bytes[6] {
            1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
            2 | 3 => {
                if bytes.len() < 12 {
                    return Err("truncated header".into());
                }
                let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
                (len as usize, 12)
            }
            v => return Err(format!("unsupported format version {v}")),
        };
        let header = bytes
            .get(start..start + header_len)
            .ok_or("truncated header")?;
        let header = std::str::from_utf8(header).map_err(|_| "header is not UTF-8")?;
        if header.contains("'fortran_order': True") {
            return Err("Fortran order is not supported".into());
        }
        let shape = parse_shape(header)?;
        let descr = header_value(header, "descr").ok_or("missing descr")?;
        let (dtype, fields, item_size) = if descr.starts_with('[') {
            let fields = parse_fields(descr)?;
            let size = fields.iter().map(|(_, d, _)| d.size()).sum();
            (None, Some(fields), size)
        } else {
            let d = descr.trim_matches('\'');
            let dtype = Dtype::parse(d).ok_or_else(|| format!("unsupported dtype {d}"))?;
            (Some(dtype), None, dtype.size())
        };
        let n: usize = shape.iter().product();
        let data = bytes[start + header_len..].to_vec();
        if data.len() != n * item_size {
            return Err(format!(
                "data length {} does not match shape {:?}",
                data.len(),
                shape
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

    /// True if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All elements of a plain array converted to `f64` (exact for every supported type).
    pub fn to_f64(&self) -> std::result::Result<Vec<f64>, String> {
        let dtype = self.dtype.ok_or("structured array: use field()")?;
        Ok(self
            .data
            .chunks_exact(self.item_size)
            .map(|b| dtype.read_f64(b))
            .collect())
    }

    /// All elements of a plain `f4` array.
    pub fn to_f32(&self) -> std::result::Result<Vec<f32>, String> {
        match self.dtype {
            Some(Dtype::F32) => Ok(self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            other => Err(format!("expected <f4, found {other:?}")),
        }
    }

    /// One field of a structured array converted to `f64`.
    pub fn field(&self, name: &str) -> std::result::Result<Vec<f64>, String> {
        let fields = self.fields.as_ref().ok_or("not a structured array")?;
        let (_, dtype, offset) = fields
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| format!("no field {name}"))?;
        Ok(self
            .data
            .chunks_exact(self.item_size)
            .map(|rec| dtype.read_f64(&rec[*offset..]))
            .collect())
    }

    /// One `f4` field of a structured array, exactly.
    pub fn field_f32(&self, name: &str) -> std::result::Result<Vec<f32>, String> {
        Ok(self.field(name)?.into_iter().map(|v| v as f32).collect())
    }
}

/// Serialise a plain array with the given shape.
pub fn to_bytes(shape: &[usize], column: &Column) -> Vec<u8> {
    let mut out = header_bytes(&format!("'{}'", column.descr()), shape);
    for i in 0..column.len() {
        column.write_element(i, &mut out);
    }
    out
}

/// Serialise a structured (record) array; all columns must have the same length.
pub fn record_to_bytes(fields: &[(&str, Column)]) -> Vec<u8> {
    let n = fields.first().map_or(0, |(_, c)| c.len());
    let descr = format!(
        "[{}]",
        fields
            .iter()
            .map(|(name, c)| format!("('{}', '{}')", name, c.descr()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut out = header_bytes(&descr, &[n]);
    for i in 0..n {
        for (_, c) in fields {
            c.write_element(i, &mut out);
        }
    }
    out
}

/// Write a plain array to `path`.
pub fn write(path: impl AsRef<Path>, shape: &[usize], column: &Column) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, to_bytes(shape, column)).map_err(|e| Error::io(path, e))
}

/// Write a structured array to `path`.
pub fn write_record(path: impl AsRef<Path>, fields: &[(&str, Column)]) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, record_to_bytes(fields)).map_err(|e| Error::io(path, e))
}

fn header_bytes(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape = match shape {
        [n] => format!("({n},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': {shape}, }}");
    // Pad so that magic (6) + version (2) + length (2) + header is a multiple of 64.
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

/// Value text of `'key': value` in the header dict (up to the matching delimiter).
fn header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(&format!("'{key}':"))? + key.len() + 3;
    let rest = header[start..].trim_start();
    let end = match rest.as_bytes().first()? {
        b'[' => matching(rest, b'[', b']')?,
        b'(' => matching(rest, b'(', b')')?,
        b'\'' => rest[1..].find('\'')? + 2,
        _ => rest.find(',')?,
    };
    Some(&rest[..end])
}

fn matching(s: &str, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0;
    for (i, b) in s.bytes().enumerate() {
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

fn parse_shape(header: &str) -> std::result::Result<Vec<usize>, String> {
    let shape = header_value(header, "shape").ok_or("missing shape")?;
    shape
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| format!("bad shape {shape}")))
        .collect()
}

fn parse_fields(descr: &str) -> std::result::Result<Vec<(String, Dtype, usize)>, String> {
    let mut fields = Vec::new();
    let mut offset = 0;
    for part in descr.trim_matches(|c| c == '[' || c == ']').split(')') {
        let part = part.trim_start_matches(|c: char| c == ',' || c.is_whitespace() || c == '(');
        if part.is_empty() {
            continue;
        }
        let mut items = part.split(',').map(|s| s.trim().trim_matches('\''));
        let name = items.next().ok_or("bad field")?.to_string();
        let d = items.next().ok_or("bad field")?;
        let dtype = Dtype::parse(d).ok_or_else(|| format!("unsupported field dtype {d}"))?;
        fields.push((name, dtype, offset));
        offset += dtype.size();
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrip() {
        let bytes = record_to_bytes(&[
            ("a", Column::F32(vec![1.5, -2.0])),
            ("b", Column::I64(vec![7, 8])),
            ("c", Column::Bool(vec![true, false])),
        ]);
        assert_eq!((bytes.len() - 2 * 13) % 64, 0);
        let a = NpyArray::from_bytes(&bytes).unwrap();
        assert_eq!(a.shape, vec![2]);
        assert_eq!(a.field_f32("a").unwrap(), vec![1.5, -2.0]);
        assert_eq!(a.field("b").unwrap(), vec![7.0, 8.0]);
        assert_eq!(a.field("c").unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn plain_2d() {
        let bytes = to_bytes(&[2, 3], &Column::I32(vec![1, 2, 3, 4, 5, 6]));
        let a = NpyArray::from_bytes(&bytes).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.to_f64().unwrap()[5], 6.0);
    }
}
