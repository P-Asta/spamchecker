use csv::ReaderBuilder;
use linfa::prelude::Predict;
use linfa::{traits::Fit, DatasetBase};
use linfa_bayes::GaussianNb;
use ndarray::{Array1, Array2};
use std::io::BufRead;

const MAX_LEN: usize = 200;

fn main() {
    let mut x = vec![];
    let mut y = vec![];

    let mut datas = ReaderBuilder::new()
        .has_headers(false)
        .from_path("data.csv")
        .unwrap();

    for data in datas.deserialize::<Data>().flatten() {
        x.push(convert(data.message));
        y.push(if data.category == *"ham" { 0 } else { 1 });
    }
    let flat_x: Vec<f32> = x.iter().flatten().map(|x| *x).collect();
    let x = Array2::from_shape_vec((x.len(), MAX_LEN), flat_x).unwrap();
    let y = Array1::from_vec(y);

    let train = DatasetBase::new(x, y);

    let model = GaussianNb::params().fit(&train).unwrap();

    //let input = std::io::stdin().lock().lines().next().unwrap().unwrap();
    //let input = "🔞 Best Free Onlyfans leaks, Teen content, porn, sexcam and daily leaks here: discord.gg/teen-sex ❤️ @here @everyone 💖WITH NEW CONTENT 💖 JOIN NOW❤️"
    let input = "come here https://discord.gg/sexx @everyone @here".to_string();
    //let input = "@everyone @here Hello World!".to_string();
    let test = Array2::from_shape_vec((1, MAX_LEN), convert(input)).unwrap();
    println!("{}", model.predict(test).targets);
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Data {
    category: String,
    message: String,
}

fn convert(message: String) -> Vec<f32> {
    let len = message.len();
    if MAX_LEN <= len {
        let mut a = message[..MAX_LEN]
            .chars()
            .map(|x| (x as u8) as f32)
            .collect::<Vec<f32>>();
        a.extend(vec![0.; MAX_LEN - a.len()]);
        a
    } else {
        let mut a = message
            .chars()
            .map(|x| (x as u8) as f32)
            .collect::<Vec<f32>>();
        a.extend(vec![0.; MAX_LEN - a.len()]);
        a
    }
}
