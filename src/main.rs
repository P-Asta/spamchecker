use csv::ReaderBuilder;
use linfa::prelude::Predict;
use linfa::{traits::Fit, DatasetBase};
use linfa_bayes::{GaussianNb, GaussianNbParams};
use ndarray::{Array1, Array2};
use std::io::BufRead;

const MAX_LEN: usize = 20;

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
        if data.category == "spam" {
            println!("{:?}", converted);
        }
    }
    let flat_x: Vec<f32> = x.iter().flatten().map(|x| *x).collect();
    let x = Array2::from_shape_vec((x.len(), MAX_LEN), flat_x).unwrap();
    let y = Array1::from_vec(y);

    let train = DatasetBase::new(x, y);

    let model: GaussianNb<_, _> = GaussianNb::params().fit(&train).unwrap();

    //let input = std::io::stdin().lock().lines().next().unwrap().unwrap();
    //let input = "if you pick the wrong key your system32 is deleted".to_string();
    let input = "
@here
Dioxus 0.5 is here!

https://www.reddit.com/r/rust/comments/1bpvy1u/dioxus_05_huge_signal_rewrite_remove_lifetimes/

This release was an absolutely monumental amount of work - equivalent to the initial release of Dioxus five times over. Across all repositories we modified over 200,000 lines of code.

With the signal rewrite, Dioxus is much much easier to work with. Between cargo templates, hotreloading, autoformatting, Copy-state, and the bundler, it's hard to express how quickly you can get an app off the ground. I encourage everyone to give the new version for a test drive.

The stuff that's changed includes:

- Complete rewrite of dioxus-core, removing all unsafe code
- Abandoning use_state and use_ref for a clone-free Signal-based API
- Removal of all lifetimes and the cx: Scope state
- A single, unified launch function that starts your app for any platform
- Asset hotreloading that supports Tailwind and Vanilla CSS
- Rewrite of events, allowing access to the native WebSys event types
- Extension of components with element properties (IE a Link now takes all of <a/> properties)
- Integrated Error Boundaries and Server Futures with Suspense integration
- 5x faster desktop reconciliation and custom asset handlers for streaming bytes
- Streaming server functions and fullstack hotreloading
- Tons of QoL improvements, bug fixes, and more!
- I want to call out the small but mighty Dioxus core team: ealmloff, dogedark, and nicoburns, as well as marc2332.

Thanks for the all the support, we're excited for the rest of 2024.
        ".to_string();
    //let input =
    //    "come here aaaaaaaaaaaaaaaaaaaaaa https://discord.gg/sexx @everyone @here".to_string();
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
        ("$", 0.8),
        ("nsfw", 2.0),
        ("who", 2.0),
        ("want", 2.0),
        ("bro", 1.0),
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
        ("@everyone", 0.2),
        ("@here", 0.4),
    ];
    let mut res = vec![];
    for spam_word in spam_words {
        res.push((have_cnt(&message, spam_word.0) as f32).powf(spam_word.1))
    }
    res
}

fn have_cnt(text: &str, c: &str) -> usize {
    let text = text
        .replace("money", "$")
        .replace(":peach:", "🍑")
        .replace(":underage:", "🔞")
        .replace("18+", "🔞");
    text.split(c).collect::<Vec<_>>().len() - 1
}
