#![feature(iterator_try_collect)]
use core::panic;
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::Display,
    hash::{Hash, Hasher},
};

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

struct Base36 {
    n: u64,
    pad: i32,
}

impl Display for Base36 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n;
        if n == 0 && self.pad <= 1 {
            return write!(f, "0");
        }
        let r = n % 36;
        let c = if r >= 10 {
            ((r - 10) + 'A' as u64) as u8 as char
        } else {
            (r + '0' as u64) as u8 as char
        };
        let d = n / 36;
        if self.pad > 1 || d != 0 {
            write!(
                f,
                "{}{c}",
                Base36 {
                    n: d,
                    pad: self.pad - 1
                }
            )?;
        } else {
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
fn base36(n: u64, pad: u32) -> String {
    format!("{}", Base36 { n, pad: pad as i32 })
}

pub fn generate_code(n: u64) -> String {
    let mut hasher = DefaultHasher::new();
    n.hash(&mut hasher);
    let n = (n << 8) | (hasher.finish() & 0xff);
    format!("{}", Base36 { n, pad: 6 })
}

fn decode_base36(s: &[u8]) -> u64 {
    let mut rv = 0u64;
    for &x in s {
        rv += if x >= '0' as u8 && x <= '9' as u8 {
            x - '0' as u8
        } else if x >= 'A' as u8 && x <= 'Z' as u8 {
            (x - 'A' as u8) + 10
        } else {
            panic!()
        } as u64;
        rv *= 36;
    }
    rv
}

fn try_decode(s: &[u8]) -> Option<u64> {
    let n = decode_base36(s);
    let h = n & 0xff;
    let n = n >> 8;
    let mut hasher = DefaultHasher::new();
    n.hash(&mut hasher);
    if hasher.finish() & 0xff == h {
        Some(n)
    } else {
        None
    }
}

pub fn decode_code(s: &str) -> Vec<u64> {
    // detect codes, check their validity and return the valid ones
    let mut rv = vec![];
    let s: Vec<u8> = s.chars().filter(|x| *x != ' ').map(|x| x as u8).collect();
    let min_code_length = 6;
    let max_code_length = 6;
    for i in 0..s.len() {
        for j in i + min_code_length..=s.len().min(i + max_code_length + 1) {
            // for all non-empty subsequences
            let s = &s[i..j];
            if let Some(n) = try_decode(s) {
                rv.push(n);
            }
        }
    }
    rv
}

#[cfg(test)]
mod test {
    use crate::base36;

    #[test]
    fn test_base36() {
        assert_eq!(&base36(0, 3), "000");
        assert_eq!(&base36(1, 3), "001");
        assert_eq!(&base36(10, 3), "00A");
        assert_eq!(&base36(10 + 36, 3), "01A");
        assert_eq!(&base36(35, 3), "00Z");
        assert_eq!(&base36(36, 3), "010");
        assert_eq!(&base36(36 * 36, 3), "100");
        assert_eq!(&base36(36 * 36 * 35, 3), "Z00");
        assert_eq!(&base36(36 * 36 * 36, 3), "1000");
    }
}
