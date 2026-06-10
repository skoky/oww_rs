use crate::RING_BUFFER_SIZE;
use crate::VOICE_SAMPLE_RATE;
use crate::chunk::Chunk;
use chrono::Utc;
use log::{debug, warn};

/// Saves chunks into a wav file (e.g. to be uploaded to cloud). All the input values are used for the
/// file name. The output file is created in `based_dir`. If storing stereo data, channels must be
/// interlaced in advance. When `short_wav` is true only the last third of the ring buffer is written.
pub fn save_full_wav(chunks: Vec<Chunk>, based_dir: String, detection_prc: u8, me: &str, prefix: &str, max_rms: i16, short_wav: bool) {
    let channels = chunks.first().unwrap().number_of_channels();
    let spec = hound::WavSpec {
        channels,
        sample_rate: VOICE_SAMPLE_RATE as _,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let now = Utc::now();
    let ts = now.format("%y%m%d-%H%M%S").to_string();

    let filename = format!("{based_dir}/{}_{}_{}_{}_{}.wav", prefix, me, detection_prc, ts, max_rms);
    let mut writer = hound::WavWriter::create(&filename, spec).unwrap();

    let save_from_index = RING_BUFFER_SIZE - (RING_BUFFER_SIZE / 3); // last 1/3

    if chunks.len() != RING_BUFFER_SIZE {
        debug!("Upload not done, ring buffer not full");
        return;
    };

    for (index, chunk) in chunks.iter().map(|c| c.to_interleaved_channels()).enumerate() {
        if !short_wav || index > save_from_index {
            for sample in chunk.iter() {
                if let Err(e) = writer.write_sample(*sample) {
                    warn!("Error writing to wav file {:?}", e);
                    break;
                }
            }
        }
    }
    debug!("Recording saved to {filename}");
}
