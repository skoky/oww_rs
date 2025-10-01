pub fn f32_to_i16(sample: &f32) -> i16 {
    // Scale to i16 range and clamp
    let scaled = sample * i16::MAX as f32;
    if scaled >= i16::MAX as f32 {
        i16::MAX
    } else if scaled <= i16::MIN as f32 {
        i16::MIN
    } else {
        scaled as i16
    }
}

pub fn i16_to_f32(sample: &i16) -> f32 {
    if *sample == i16::MIN { -1.0 } else { *sample as f32 / i16::MAX as f32 }
}

// pub fn u16_to_f32(sample: &u16) -> f32 {
//     (*sample as i32 - i16::MAX as i32) as f32 / i16::MAX as f32
// }

// pub fn stereo_to_mono(stereo_data: &[f32]) -> Vec<f32> {
//     if stereo_data.len() % 2 != 0 {
//         error!("Stereo data must contain an even number of samples");
//         return stereo_data.to_vec();
//     }
//
//     let num_frames = stereo_data.len() / 2;
//
//     let mut mono_data = Vec::with_capacity(num_frames);
//
//     for frame in 0..num_frames {
//         let left_sample = stereo_data[frame * 2];
//         let right_sample = stereo_data[frame * 2 + 1];
//
//         // Average the left and right channels
//         let mono_sample = (left_sample + right_sample) * 0.5;
//         mono_data.push(mono_sample);
//     }
//
//     mono_data
// }

mod tests {
    use crate::mic::converters::{f32_to_i16, i16_to_f32};

    #[test]
    fn test_conversion() {
        use approx::assert_relative_eq;
        let i = vec![0i16, 14, 31, 22, 150, -256, -1i16, i16::MIN, i16::MAX];
        let f: Vec<f32> = i.iter().map(i16_to_f32).collect();
        let ii: Vec<i16> = f.iter().map(f32_to_i16).collect();
        for (x, y) in ii.iter().zip(i) {
            assert_relative_eq!(*x as f32, y as f32, epsilon = 1.0);
        }
    }
}
