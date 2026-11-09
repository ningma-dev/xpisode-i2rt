//! Torques for cases in a file, for comparing with another implementation.

use xpisode_i2rt::config::Link;
use xpisode_i2rt::dynamics::Chain;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    links: Vec<Link>,
    gravity: [f64; 3],
    cases: Vec<Vec<f64>>,
}

fn main() {
    let text = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    let input: Input = serde_json::from_str(&text).unwrap();
    let chain = Chain::new(&input.links, input.gravity);
    let n = chain.len();
    let out: Vec<Vec<f64>> = input
        .cases
        .iter()
        .map(|c| {
            let mut tau = vec![0.0; n];
            chain.inverse_dynamics(&c[..n], &c[n..2 * n], &c[2 * n..3 * n], true, &mut tau);
            tau
        })
        .collect();
    println!("{}", serde_json::to_string(&out).unwrap());
}
