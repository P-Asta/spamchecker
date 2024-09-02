use csv::ReaderBuilder;
use linfa::prelude::Predict;
use linfa::{traits::Fit, DatasetBase};
use linfa_bayes::{GaussianNb, GaussianNbParams};
use ndarray::{Array1, Array2};
use std::io::BufRead;

const MAX_LEN: usize = 25;

fn main() {
    let mut x = vec![];
    let mut y = vec![];

    let mut datas = ReaderBuilder::new()
        .has_headers(false)
        .from_path("data.csv")
        .unwrap();

    for data in datas.deserialize::<Data>().flatten() {
        let converted = convert(data.message);
        x.push(converted.clone());
        y.push(if data.category == *"ham" { 0 } else { 1 });
    }
    let flat_x: Vec<f32> = x.iter().flatten().map(|x| *x).collect();
    let x = Array2::from_shape_vec((x.len(), MAX_LEN), flat_x).unwrap();
    let y = Array1::from_vec(y);

    let train = DatasetBase::new(x, y);

    let model: GaussianNb<_, _> = GaussianNb::params().fit(&train).unwrap();

    //let input = std::io::stdin().lock().lines().next().unwrap().unwrap();

    //let input = "
    //18+ Teen Girls and onlyfans leaks for free :peach: :underage:  https://discord.gg/tiktokhomes @everyone
    //".to_string();

    //    let input = "
    //    @everyone @here
    //            🔥 Burn 0.14.0 release <:EmberAmazed:1184985554014715935> @everyone
    //
    //This release marks the debut of our CubeCL integration, which brings cross-platform GPU programming capabilities directly to Rust. With CubeCL now supporting both CUDA and WebGPU, Burn benefits from a new CUDA backend that can be enabled using the `cuda-jit` feature. Please note that this backend is still considered experimental, and some operations, particularly those related to vision, may experience issues.
    //
    //Burn 0.14.0 introduces a new tensor data format that significantly enhances serialization and deserialization speeds and introduces Quantization, a new Beta feature included in this release. The format is not compatible with previous versions of Burn, but you can migrate your previously saved records using [this guide](https://github.com/tracel-ai/burn?tab=readme-ov-file#deprecation).
    //
    //Thanks to all contributors, over 50 for this release alone 🎉
    //
    //There's a lot more in this release! I recommend taking a look at the full release notes.
    //
    //Release Notes: https://github.com/tracel-ai/burn/releases/tag/v0.14.0
    //Reddit Post: https://www.reddit.com/r/rust/comments/1f2n1iq/burn_0140_released_the_first_fully_rustnative/
    //            ".to_string();

    let input =
        "come here aaaaaaaaaaaaaaaaaaaaaa https://discord.gg/sexx @everyone @here".to_string();
    //let input = "@everyone ny64 is furry".to_string();
    let test = Array2::from_shape_vec((1, MAX_LEN), convert(input)).unwrap();
    println!(
        "it is {}",
        if model.predict(test).targets[0] == 0 {
            "ham"
        } else {
            "spam"
        }
    );
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Data {
    category: String,
    message: String,
}

fn convert(message: String) -> Vec<f32> {
    let spam_words: [(&str, f32); MAX_LEN] = [
        ("now", 1.0),
        ("announced", 1.0),
        ("release", 1.0),
        ("$", 0.8),
        ("nsfw", 2.0),
        ("who", 1.2),
        ("want", 1.2),
        ("join", 1.0),
        ("bro", 1.0),
        ("you", 1.0),
        ("teen", 0.8),
        ("onlyfans", 2.0),
        ("leaks", 2.0),
        ("free", 1.2),
        ("nudes", 2.0),
        ("porn", 2.0),
        ("🍑", 1.0),
        ("🔞", 1.0),
        ("[", 0.8),
        ("]", 0.8),
        ("(", 0.8),
        (")", 0.8),
        ("sex", 2.0),
        ("@everyone", 0.8),
        ("@here", 0.4),
    ];
    let mut res = vec![];
    for spam_word in spam_words {
        res.push((have_cnt(&message, spam_word.0) as f32 + 0.1).powf(spam_word.1))
    }
    println!("{res:?}");
    res
}

fn have_cnt(text: &str, c: &str) -> usize {
    let mut message = "_".to_string();
    message.push_str(text);
    message.push('_');
    let text = text
        .replace("money", "$")
        .replace(":peach:", "🍑")
        .replace(":underage:", "🔞")
        .replace("18+", "🔞");
    text.split(c).collect::<Vec<_>>().len() - 1
}
