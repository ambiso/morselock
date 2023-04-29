#![feature(iterator_try_collect)]
use std::collections::HashMap;

use lazy_static::lazy_static;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum MorseSymbol {
    Dot,
    Dash,
}

lazy_static! {
    static ref DECODE_MORSE_MAP: HashMap<&'static [MorseSymbol], char> = {
        use MorseSymbol::*;
        let mut m = HashMap::new();
        m.insert([Dot, Dash].as_slice(), 'A');
        m.insert([Dash, Dot, Dot, Dot].as_slice(), 'B');
        m.insert([Dash, Dot, Dash, Dot].as_slice(), 'C');
        m.insert([Dash, Dot, Dot].as_slice(), 'D');
        m.insert([Dot].as_slice(), 'E');
        m.insert([Dot, Dot, Dash, Dot].as_slice(), 'F');
        m.insert([Dash, Dash, Dot].as_slice(), 'G');
        m.insert([Dot, Dot, Dot, Dot].as_slice(), 'H');
        m.insert([Dot, Dot].as_slice(), 'I');
        m.insert([Dot, Dash, Dash, Dash].as_slice(), 'J');
        m.insert([Dash, Dot, Dash].as_slice(), 'K');
        m.insert([Dot, Dash, Dot, Dot].as_slice(), 'L');
        m.insert([Dash, Dash].as_slice(), 'M');
        m.insert([Dash, Dot].as_slice(), 'N');
        m.insert([Dash, Dash, Dash].as_slice(), 'O');
        m.insert([Dot, Dash, Dash, Dot].as_slice(), 'P');
        m.insert([Dash, Dash, Dot, Dash].as_slice(), 'Q');
        m.insert([Dot, Dash, Dot].as_slice(), 'R');
        m.insert([Dot, Dot, Dot].as_slice(), 'S');
        m.insert([Dash].as_slice(), 'T');
        m.insert([Dot, Dot, Dash].as_slice(), 'U');
        m.insert([Dot, Dot, Dot, Dash].as_slice(), 'V');
        m.insert([Dot, Dash, Dash].as_slice(), 'W');
        m.insert([Dash, Dot, Dot, Dash].as_slice(), 'X');
        m.insert([Dash, Dot, Dash, Dash].as_slice(), 'Y');
        m.insert([Dash, Dash, Dot, Dot].as_slice(), 'Z');
        m.insert([Dot, Dash, Dash, Dash, Dash].as_slice(), '1');
        m.insert([Dot, Dot, Dash, Dash, Dash].as_slice(), '2');
        m.insert([Dot, Dot, Dot, Dash, Dash].as_slice(), '3');
        m.insert([Dot, Dot, Dot, Dot, Dash].as_slice(), '4');
        m.insert([Dot, Dot, Dot, Dot, Dot].as_slice(), '5');
        m.insert([Dash, Dot, Dot, Dot, Dot].as_slice(), '6');
        m.insert([Dash, Dash, Dot, Dot, Dot].as_slice(), '7');
        m.insert([Dash, Dash, Dash, Dot, Dot].as_slice(), '8');
        m.insert([Dash, Dash, Dash, Dash, Dot].as_slice(), '9');
        m.insert([Dash, Dash, Dash, Dash, Dash].as_slice(), '0');
        m
    };
    static ref ENCODE_MORSE_MAP: HashMap<char, &'static [MorseSymbol]> = {
        let mut m = HashMap::new();
        for (k, v) in DECODE_MORSE_MAP.iter() {
            m.insert(*v, *k);
        }
        m
    };
}

pub fn decode_morse_symbol_sequence(sequence: &[MorseSymbol]) -> Option<char> {
    DECODE_MORSE_MAP.get(sequence).copied()
}

pub fn encode_letter(ch: char) -> Option<&'static [MorseSymbol]> {
    ENCODE_MORSE_MAP.get(&ch).copied()
}

pub fn k_means_clustering<const N: usize>(
    starting_means: Option<[i32; N]>,
    data: &[i32],
) -> [i32; N] {
    let mut means = if let Some(starting_means) = starting_means {
        starting_means
    } else {
        let mean = (data.iter().map(|x| *x as i64).sum::<i64>() / data.len() as i64) as i32;
        let mut means = [0i32; N];
        for (i, x) in means.iter_mut().enumerate() {
            *x = mean + i as i32 - (N as i32 / 2);
        }
        means
    };

    loop {
        let mut sums = [0i64; N];
        let mut counters = [0i32; N];
        for x in data {
            // get the mean with the smallest distance to this data point
            let (i, _) = means
                .iter()
                .map(|m| (m - x).abs())
                .enumerate()
                .min_by_key(|x| x.1)
                .unwrap();
            // increment the sum of that mean by the datapoint (so we can later compute the new mean of the data points closer to this mean)
            sums[i] += *x as i64;
            counters[i] += 1;
        }

        let mut new_means = [0; N];
        for (i, (&s, &c)) in sums.iter().zip(counters.iter()).enumerate() {
            if c != 0 {
                new_means[i] = (s / c as i64) as i32;
            }
        }

        if new_means == means {
            break;
        }
        means = new_means;
    }

    means.sort();

    means
}

#[cfg(test)]
mod tests {
    use crate::k_means_clustering;

    #[test]
    fn test_clustering() {
        assert_eq!(
            k_means_clustering(None, &[10, 10, 10, 20, 20, 20]),
            [10, 20]
        );
        assert_eq!(
            k_means_clustering(None, &[10, 10, 10, 10, 10, 10, 20, 20, 20]),
            [10, 20]
        );
        assert_eq!(
            k_means_clustering(None, &[9, 9, 10, 10, 10, 10, 11, 11, 19, 20, 21]),
            [10, 20]
        );
        assert_eq!(k_means_clustering(None, &[1, 2, 3]), [1, 2, 3]);
        assert_eq!(
            k_means_clustering(None, &[10, 10, 10, 20, 20, 20, 30, 30, 30]),
            [10, 20, 30]
        );
    }
}
