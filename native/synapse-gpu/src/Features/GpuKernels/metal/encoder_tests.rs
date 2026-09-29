use super::metal_matvec_rows;
use crate::gpu_kernels::decoder::Encoding;

#[test]
fn matvec_rows_match_the_shader_instances() {
    let shader = include_str!("shaders/matmul.metal");
    let instances: Vec<(u32, u32)> = shader
        .lines()
        .filter_map(|line| {
            let (tokens, rows) = line
                .strip_prefix("SYNAPSE_MATVEC(")?
                .strip_suffix(')')?
                .split_once(", ")?;
            Some((tokens.parse().ok()?, rows.parse().ok()?))
        })
        .collect();

    assert_eq!(instances.len(), 4);
    for (tokens, rows) in instances {
        assert_eq!(metal_matvec_rows(Encoding::Q8_0, tokens), rows);
    }
}

#[test]
fn kquant_matvec_rows_match_the_shader_instances() {
    let shader = include_str!("shaders/matmul_kquant.metal");
    let instances: Vec<(Encoding, u32, u32)> = shader
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line
                .strip_prefix("SYNAPSE_KQ_MATVEC(")?
                .strip_suffix(')')?
                .split(", ")
                .collect();
            let encoding = if *fields.get(1)? == "SYNAPSE_Q4K" {
                Encoding::Q4K
            } else {
                Encoding::Q6K
            };
            Some((
                encoding,
                fields.get(2)?.parse().ok()?,
                fields.get(3)?.parse().ok()?,
            ))
        })
        .collect();

    assert_eq!(instances.len(), 8);
    for (encoding, tokens, rows) in instances {
        // Four simdgroups of R rows each.
        assert_eq!(metal_matvec_rows(encoding, tokens), 4 * rows);
    }
}
