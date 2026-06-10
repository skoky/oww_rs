use log::info;
use nnnoiseless::{DenoiseState, FRAME_SIZE};
use std::collections::VecDeque;

/// Denoiser helper using `nnnoiseless` with the default model embedded in the crate.
/// When disabled it passes audio through unchanged.
pub struct AudioDenoiser {
    enabled: bool,
    denoiser: Box<DenoiseState<'static>>,
    /// single frame reusable buffer
    buffer: [f32; DenoiseState::FRAME_SIZE],
    /// queue of samples holding leftovers when the input chunk is not a multiple of [`FRAME_SIZE`]
    queue: VecDeque<f32>,
}

impl AudioDenoiser {
    pub fn new(enable_denoising: bool) -> Self {
        if enable_denoising {
            info!("Denoiser activated")
        }
        Self {
            enabled: enable_denoising,
            denoiser: DenoiseState::new(),
            buffer: [0.0; DenoiseState::FRAME_SIZE],
            queue: VecDeque::new(),
        }
    }

    pub fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        // returns original content if disabled
        if !self.enabled {
            return chunk.to_vec();
        };

        chunk.iter().for_each(|x| self.queue.push_back(*x));

        // loop over data if bigger than the FRAME_SIZE
        let mut output: Vec<f32> = vec![];
        while self.queue.len() > FRAME_SIZE {
            let data = self.queue.drain(..FRAME_SIZE).collect::<Vec<_>>();
            self.denoiser.process_frame(&mut self.buffer, &data);
            self.buffer.iter().for_each(|x| output.push(*x));
        }

        output
    }
}
