use synapse_kernels::cpu_kernels::{
    ABI_VERSION, BLOCK_BYTES, BLOCK_ELEMENTS, Isa, KernelError, Q8Activations, Q8Weights,
    STATUS_INVALID_SHAPE, STATUS_NULL_POINTER, STATUS_OK, fp16_to_f32, matmul, matmul_with,
    synapse_kernels_abi_version, synapse_kernels_capabilities, synapse_q8_0_matmul,
};

const SENTINEL: f32 = 12_345.0;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 33).expect("31-bit value")
    }

    fn code(&mut self) -> i8 {
        i8::try_from(i64::from(self.next() % 255) - 127).expect("code in range")
    }
}

struct Case {
    rows: usize,
    blocks: usize,
    tokens: usize,
    stride: usize,
    weights: Vec<u8>,
    quants: Vec<i8>,
    scales: Vec<f32>,
}

impl Case {
    fn new(rows: usize, blocks: usize, tokens: usize, stride: usize, seed: u64) -> Self {
        let mut random = Lcg(seed);
        let mut weights = Vec::with_capacity(rows * blocks * BLOCK_BYTES);
        for _ in 0..rows * blocks {
            let scale = u16::try_from(0x1000 + random.next() % 0x1400).expect("fp16 bits");
            weights.extend_from_slice(&scale.to_le_bytes());
            weights.extend((0..BLOCK_ELEMENTS).map(|_| random.code().to_le_bytes()[0]));
        }

        let quants = (0..tokens * blocks * BLOCK_ELEMENTS)
            .map(|_| random.code())
            .collect();
        let scales = (0..tokens * blocks)
            .map(|_| {
                fp16_to_f32(u16::try_from(0x2000 + random.next() % 0x1000).expect("fp16 bits"))
            })
            .collect();
        Self {
            rows,
            blocks,
            tokens,
            stride,
            weights,
            quants,
            scales,
        }
    }

    const fn weights(&self) -> Q8Weights<'_> {
        Q8Weights {
            data: self.weights.as_slice(),
            rows: self.rows,
            blocks_per_row: self.blocks,
        }
    }

    const fn activations(&self) -> Q8Activations<'_> {
        Q8Activations {
            quants: self.quants.as_slice(),
            scales: self.scales.as_slice(),
            tokens: self.tokens,
        }
    }

    fn output(&self) -> Vec<f32> {
        vec![SENTINEL; (self.tokens - 1) * self.stride + self.rows]
    }

    fn exact(&self, row: usize, token: usize) -> (f64, f64) {
        let mut value = 0.0;
        let mut magnitude = 0.0;
        for block in 0..self.blocks {
            let start = (row * self.blocks + block) * BLOCK_BYTES;
            let weight_scale = f64::from(fp16_to_f32(u16::from_le_bytes([
                self.weights[start],
                self.weights[start + 1],
            ])));
            let codes =
                &self.quants[(token * self.blocks + block) * BLOCK_ELEMENTS..][..BLOCK_ELEMENTS];
            let dot: i64 = self.weights[start + 2..start + BLOCK_BYTES]
                .iter()
                .zip(codes)
                .map(|(&weight, &code)| i64::from(i8::from_le_bytes([weight])) * i64::from(code))
                .sum();
            let term = weight_scale
                * f64::from(self.scales[token * self.blocks + block])
                * f64::from(i32::try_from(dot).expect("block dot"));
            value += term;
            magnitude += term.abs();
        }

        (value, magnitude)
    }

    fn assert_matches_oracle(&self, output: &[f32], label: &str) {
        for token in 0..self.tokens {
            for row in 0..self.rows {
                let (exact, magnitude) = self.exact(row, token);
                let actual = f64::from(output[token * self.stride + row]);
                assert!(
                    (actual - exact).abs() <= 2e-5f64.mul_add(magnitude, 1e-12),
                    "{label}: row {row} token {token}: {actual} vs {exact}"
                );
            }

            if token + 1 < self.tokens {
                assert!(
                    output[token * self.stride + self.rows..(token + 1) * self.stride]
                        .iter()
                        .all(|value| value.to_bits() == SENTINEL.to_bits()),
                    "{label}: stride gap was written"
                );
            }
        }
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case::new(37, 28, 1, 37, 1),
        Case::new(16, 152, 3, 20, 2),
        Case::new(5, 64, 2, 9, 3),
        Case::new(8, 63, 4, 11, 4),
    ]
}

#[test]
fn every_supported_isa_matches_f64_oracle() {
    for isa in [Isa::Scalar, Isa::NeonDotProd, Isa::Avx2]
        .into_iter()
        .filter(|isa| isa.is_supported())
    {
        for case in cases() {
            let mut output = case.output();
            matmul_with(
                isa,
                &case.weights(),
                &case.activations(),
                &mut output,
                case.stride,
            )
            .expect("valid case");
            case.assert_matches_oracle(
                &output,
                &format!("{isa:?} {}x{}x{}", case.rows, case.blocks, case.tokens),
            );
        }
    }

    assert!(Isa::detect().is_supported());
}

#[test]
fn invalid_shapes_are_rejected_before_reading() {
    let case = Case::new(4, 2, 2, 4, 9);
    let mut output = case.output();
    let empty = Q8Weights {
        rows: 0,
        ..case.weights()
    };
    assert_eq!(
        matmul(&empty, &case.activations(), &mut output, 4),
        Err(KernelError::InvalidShape)
    );
    assert_eq!(
        matmul(&case.weights(), &case.activations(), &mut output, 3),
        Err(KernelError::InvalidShape)
    );
    let short = Q8Weights {
        data: &case.weights[..case.weights.len() - 1],
        ..case.weights()
    };
    assert_eq!(
        matmul(&short, &case.activations(), &mut output, 4),
        Err(KernelError::BufferTooSmall)
    );
    let huge = Q8Weights {
        rows: usize::MAX,
        ..case.weights()
    };
    assert_eq!(
        matmul(&huge, &case.activations(), &mut output, usize::MAX),
        Err(KernelError::Overflow)
    );
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
}

#[test]
fn unsupported_isa_is_rejected() {
    let case = Case::new(4, 2, 1, 4, 10);
    let mut output = case.output();
    for isa in [Isa::NeonDotProd, Isa::Avx2]
        .into_iter()
        .filter(|isa| !isa.is_supported())
    {
        assert_eq!(
            matmul_with(isa, &case.weights(), &case.activations(), &mut output, 4),
            Err(KernelError::UnsupportedIsa)
        );
    }
}

#[test]
fn fp16_conversion_is_exact_for_finite_values() {
    for bits in 0..=u16::MAX {
        let exponent = i32::from((bits >> 10) & 0x1f);
        if exponent == 0x1f {
            continue;
        }

        let mantissa = f64::from(bits & 0x3ff);
        let magnitude = if exponent == 0 {
            mantissa * 2f64.powi(-24)
        } else {
            (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15)
        };
        let expected = if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        };
        assert_eq!(
            f64::from(fp16_to_f32(bits)).to_bits(),
            expected.to_bits(),
            "bits {bits:#06x}"
        );
    }
}

#[test]
fn c_abi_reports_version_and_capabilities() {
    assert_eq!(synapse_kernels_abi_version(), ABI_VERSION);
    assert_eq!(
        synapse_kernels_capabilities(),
        Isa::detect().capability_bit()
    );
}

#[test]
fn c_abi_rejects_null_and_inconsistent_arguments() {
    let case = Case::new(4, 2, 1, 4, 11);
    let mut output = case.output();
    // SAFETY: the kernel returns before reading when a pointer is null or the shape is inconsistent.
    let (null_status, shape_status) = unsafe {
        (
            synapse_q8_0_matmul(
                std::ptr::null(),
                68,
                4,
                2,
                case.quants.as_ptr(),
                case.scales.as_ptr(),
                1,
                output.as_mut_ptr(),
                4,
            ),
            synapse_q8_0_matmul(
                case.weights.as_ptr(),
                67,
                4,
                2,
                case.quants.as_ptr(),
                case.scales.as_ptr(),
                1,
                output.as_mut_ptr(),
                4,
            ),
        )
    };
    assert_eq!(null_status, STATUS_NULL_POINTER);
    assert_eq!(shape_status, STATUS_INVALID_SHAPE);
}

#[test]
fn c_abi_matches_safe_api() {
    let case = Case::new(12, 28, 3, 15, 12);
    let mut expected = case.output();
    matmul(
        &case.weights(),
        &case.activations(),
        &mut expected,
        case.stride,
    )
    .expect("valid case");
    let mut actual = case.output();
    // SAFETY: every buffer in `case` covers the declared rows, blocks, tokens, and stride.
    let status = unsafe {
        synapse_q8_0_matmul(
            case.weights.as_ptr(),
            case.blocks * BLOCK_BYTES,
            case.rows,
            case.blocks,
            case.quants.as_ptr(),
            case.scales.as_ptr(),
            case.tokens,
            actual.as_mut_ptr(),
            case.stride,
        )
    };
    assert_eq!(status, STATUS_OK);
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
