//! Bundled model integrity and tokenizer boundary checks.
use matchmaker_rs::dp::otm_model::{OtmModel, tokenize};

#[test]
fn token_slots_and_bounds_match_reference() {
    let tokens = tokenize(&[vec![20, 60, 109, 61, 62, 63, 64, 65]], &[60, 110]).unwrap();
    assert_eq!(tokens[0], [40; 7]);
    assert_eq!(tokens[1], [90; 7]);
    assert_eq!(tokens[2], [89; 7]);
    assert_eq!(tokens[3], [0, 40, 0, 41, 42, 43, 44]);
    assert_eq!(tokens[25], [90; 7]);
    assert!(tokenize(&[], &[19]).is_err());
    assert!(tokenize(&[], &[111]).is_err());
    assert!(tokenize(&vec![vec![]; 26], &[]).is_err());
    assert!(tokenize(&vec![vec![]; 17], &[]).is_err());
    assert!(tokenize(&[], &[60; 9]).is_err());
    let full = tokenize(&vec![vec![60]; 16], &[60; 8]).unwrap();
    assert_eq!(full[8], [89; 7]);
    assert_eq!(full[24], [40, 0, 0, 0, 0, 0, 0]);
    assert_eq!(full[25], [90; 7]);
}

#[test]
fn checkpoint_validation_rejects_corruption() {
    let bytes = include_bytes!("../assets/otm/model.bin");
    let manifest = include_str!("../assets/otm/model.json");
    let mut broken = bytes.to_vec();
    broken[0] ^= 1;
    assert!(OtmModel::from_bytes(&broken, manifest).is_err());
    assert!(OtmModel::from_bytes(bytes, "{}").is_err());
    assert!(OtmModel::from_bytes(&bytes[..bytes.len() - 4], manifest).is_err());
    let mut metadata: serde_json::Value = serde_json::from_str(manifest).unwrap();
    metadata["version"] = 2.into();
    assert!(OtmModel::from_bytes(bytes, &metadata.to_string()).is_err());
    metadata["version"] = 1.into();
    metadata["tensors"][0]["shape"][0] = 25.into();
    assert!(OtmModel::from_bytes(bytes, &metadata.to_string()).is_err());
    metadata["tensors"][0]["shape"][0] = 26.into();
    metadata["tensors"][0]["offset"] = 4.into();
    assert!(OtmModel::from_bytes(bytes, &metadata.to_string()).is_err());
}

#[test]
fn logits_match_torch() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("golden/otm_model_layers/fixture.json")).unwrap();
    let mut tokens = [[0; 7]; 26];
    for (row, input) in tokens.iter_mut().zip(fixture["tokens"].as_array().unwrap()) {
        for (value, input) in row.iter_mut().zip(input.as_array().unwrap()) {
            *value = input.as_u64().unwrap() as usize;
        }
    }
    let actual = OtmModel::bundled().unwrap().forward(&tokens).unwrap();
    let expected = fixture["logits"].as_array().unwrap();
    for (i, (&a, e)) in actual.iter().zip(expected).enumerate() {
        let e = e.as_f64().unwrap() as f32;
        assert!((a - e).abs() < 3e-4, "logit {i}: {a} != {e}");
    }
    tokens[0][0] = 91;
    assert!(OtmModel::bundled().unwrap().forward(&tokens).is_err());
}
