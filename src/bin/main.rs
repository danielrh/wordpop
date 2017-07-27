extern crate rand;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{Read,ErrorKind,Result, Write, BufWriter};
use std::io;
use std::env;
use std::cmp;


#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word4(pub [u8;4]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word5(pub [u8;5]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word6(pub [u8;6]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word7(pub [u8;7]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word8(pub [u8;8]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word9(pub [u8;9]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word10(pub [u8;10]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word11(pub [u8;11]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word12(pub [u8;12]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word13(pub [u8;13]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word14(pub [u8;14]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word15(pub [u8;15]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word16(pub [u8;16]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word17(pub [u8;17]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word18(pub [u8;18]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word19(pub [u8;19]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word20(pub [u8;20]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word21(pub [u8;21]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word22(pub [u8;22]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word23(pub [u8;23]);
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
struct Word24(pub [u8;24]);

trait Sliceable {
    fn slice<'a>(&'a self) -> &'a [u8];
    fn slice_mut<'a> (&'a mut self) -> &'a mut [u8];
}

macro_rules! SlicableTrait {
    ($typ: tt, $num: expr) => {
        impl Sliceable for $typ {
            fn slice<'a>(&'a self) -> &'a [u8] {
                &self.0
            }
            fn slice_mut<'a> (&'a mut self) -> &'a mut [u8] {
                &mut self.0
            }
        }
        impl PartialOrd for $typ {
            fn partial_cmp(&self, other:&Self) -> Option<cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }        
         }
        impl Ord for $typ {
            fn cmp(&self, other:&Self) -> cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }
        impl Default for $typ {
            fn default() ->Self {
                $typ([0;$num])
            }
        }
    }
}

SlicableTrait!(Word4,4);
SlicableTrait!(Word5,5);
SlicableTrait!(Word6,6);
SlicableTrait!(Word7,7);
SlicableTrait!(Word8,8);
SlicableTrait!(Word9,9);
SlicableTrait!(Word10,10);
SlicableTrait!(Word11,11);
SlicableTrait!(Word12,12);
SlicableTrait!(Word13,13);
SlicableTrait!(Word14,14);
SlicableTrait!(Word15,15);
SlicableTrait!(Word16,16);
SlicableTrait!(Word17,17);
SlicableTrait!(Word18,18);
SlicableTrait!(Word19,19);
SlicableTrait!(Word20,20);
SlicableTrait!(Word21,21);
SlicableTrait!(Word22,22);
SlicableTrait!(Word23,23);
SlicableTrait!(Word24,24);

#[derive(Copy,Clone,PartialEq,Eq,Debug)]
struct Entry<Word:Sliceable+Ord+Clone+Copy> {
    internal_count: i32,
    word: Word,
    block_id:u8
}
impl<Word:Sliceable+Ord+Clone+Copy> Entry<Word> {
    fn new(word:&Word,block_id:u8, is_random_insertion:bool) ->Self {
        Entry::<Word> {
            internal_count: if is_random_insertion {-1} else {1},
            word:word.clone(),
            block_id:block_id,
        }
    }
    fn count(&self) -> i32 {
        self.internal_count.abs()
    }
    fn increment(&mut self, delta: i32) {
        if self.internal_count < 0 {
            self.internal_count -= delta;
        } else {
            self.internal_count += delta;
        }
    }
    fn reset_count_to_one(&mut self) {
        self.internal_count = 1;
    }
    fn set_nonrandom_flag(&mut self) {
        self.internal_count = self.internal_count.abs();
    }
    fn is_random_insert(&self) -> bool {
        self.internal_count < 0
    }
}
impl<Word:Sliceable+Ord+Copy> PartialOrd for Entry<Word> {
    fn partial_cmp(&self, other:&Self) -> Option<cmp::Ordering> {
        match self.word.partial_cmp(&other.word) {
            Some(val) =>  {
                match val {
                    cmp::Ordering::Less => Some(cmp::Ordering::Less),
                    cmp::Ordering::Greater => Some(cmp::Ordering::Greater),
                    cmp::Ordering::Equal => self.block_id.partial_cmp(&other.block_id),
                }
            }
            None => None
        }
    }
}
impl<Word:Sliceable+Ord+Copy> Ord for Entry<Word> {
    fn cmp(&self, other:&Self) -> cmp::Ordering {
        match self.word.cmp(&other.word) {
            cmp::Ordering::Less => cmp::Ordering::Less,
            cmp::Ordering::Greater => cmp::Ordering::Greater,
            cmp::Ordering::Equal => self.block_id.cmp(&other.block_id),
        }
    }
}

fn init_silly_rand() -> rand::ChaChaRng {
    rand::ChaChaRng::from_seed(&[((18446744073709551557u64 >> 16) >> 16) as u32, (18446744073709551557u64 & 0xffffffff) as u32][..])
}
fn silly_rand(state:&mut rand::ChaChaRng, low: u64, high: u64) -> u64{
    return state.gen_range(low, high);
}

fn find_candidate<Word:Sliceable+Ord+Default+Copy>(lru:&Vec<Entry<Word>>,
                                                   rand_state:&mut rand::ChaChaRng,
                                                   num_candidates:usize) -> usize {
    let mut best_index = silly_rand(rand_state, 0, lru.len() as u64) as usize;
    let mut best_count = lru[best_index].count();
    for _ in 0..num_candidates {
        let index = silly_rand(rand_state, 0, lru.len() as u64) as usize;
        let count = lru[best_index].count();
        if count < best_count {
            best_count = count;
            best_index = index;
        }
    }
    best_index
}

struct LRU<Word:Sliceable+Ord+Copy> {
    lru:Vec<Entry<Word>>,
    cur:Vec<Entry<Word>>,
    cur_index: usize,
    rand_state: rand::ChaChaRng,
}

#[cfg(not(feature="ascii"))]
const NTH:[u8;16] = ['0' as u8,'1' as u8,'2' as u8,'3' as u8,'4' as u8,'5' as u8,
             '6' as u8,'7' as u8,'8' as u8,'9' as u8,'a' as u8,'b' as u8,
             'c' as u8,'d' as u8,'e' as u8,'f' as u8];

#[cfg(not(feature="ascii"))]
fn nibble_to_hex(nib:u8) -> u8 {
    NTH[nib as usize]
}
fn print_encode_hex<Word:Sliceable>(buf:&mut [u8;512], word:&Word) -> usize {
    for (index, item) in word.slice().iter().enumerate() {
        buf[index<<1] = nibble_to_hex(*item>>4);
        buf[(index<<1) + 1] = nibble_to_hex(*item&0xf);
    }
    word.slice().len()<<1
}

fn print_encode_ascii<Word:Sliceable>(buf:&mut [u8;512], word:&Word) -> usize {
    buf[..word.slice().len()].clone_from_slice(word.slice());
    word.slice().len()
}

fn ends_with_constant<Word:Sliceable>(w:&Word) -> bool {
    let s = w.slice();
    let len = s.len();
    s[len - 1] == s[len - 2] &&
        s[len - 2] == s[len - 3] && 
        s[len - 3] == s[len - 4]
}

fn starts_with_constant<Word:Sliceable>(w:&Word) -> bool {
    let s = w.slice();
    s[0] == s[1] && s[1] == s[2] && s[2] == s[3]
}
#[cfg(test)]
mod test {
    use super::LRU;
    use super::Word4;
    use super::Entry;
    use super::Sliceable;
    #[test]
    fn test_eliminate_duplicates() {
        let mut dups = LRU::<super::Word4>::new(32);
        let mapping:[u8;32] = [
            1,1,1,1,
            2,3,3,3,
            3,3,3,4,
            5,5,6,6,

            7,7,7,7,
            8,8,9,9,
            10,10,10,11,
            11,12,13,14
        ];
        let block_ids:[u8;32]  = [
            0,0,1,1,
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,
            
            13,13,13,13,
            14,15,16,17,
            18,19,20,21,
            22,23,24,25
            ];
        for (index, entry) in dups.cur.iter_mut().enumerate() {
            entry.word.slice_mut()[0] = mapping[index];
            entry.block_id = block_ids[index];
            entry.internal_count = -2;
        }
        dups.cur_index = 32;
        dups.eliminate_duplicates();
        assert_eq!(&dups.cur[..14],
                   &[Entry::<Word4>{internal_count:-4,word:Word4([1,0,0,0]), block_id:1},
                   Entry::<Word4>{internal_count:-2,word:Word4([2,0,0,0]), block_id:1},
                   Entry::<Word4>{internal_count:-12,word:Word4([3,0,0,0]), block_id:7},
                   Entry::<Word4>{internal_count:-2,word:Word4([4,0,0,0]), block_id:8},
                   Entry::<Word4>{internal_count:-4,word:Word4([5,0,0,0]), block_id:10},
                   Entry::<Word4>{internal_count:-4,word:Word4([6,0,0,0]), block_id:12},
                   Entry::<Word4>{internal_count:-2,word:Word4([7,0,0,0]), block_id:13},
                   Entry::<Word4>{internal_count:-4,word:Word4([8,0,0,0]), block_id:15},
                   Entry::<Word4>{internal_count:-4,word:Word4([9,0,0,0]), block_id:17},
                   Entry::<Word4>{internal_count:-6,word:Word4([10,0,0,0]), block_id:20},
                   Entry::<Word4>{internal_count:-4,word:Word4([11,0,0,0]), block_id:22},
                   Entry::<Word4>{internal_count:-2,word:Word4([12,0,0,0]), block_id:23},
                   Entry::<Word4>{internal_count:-2,word:Word4([13,0,0,0]), block_id:24},
                   Entry::<Word4>{internal_count:-2,word:Word4([14,0,0,0]), block_id:25}][..]);
        assert_eq!(dups.cur_index,
                   14);
    }
}

impl<Word:Sliceable+Ord+Default+Copy> LRU<Word> {
    fn new(num_entries: usize) -> Self {
        LRU::<Word>{
            lru:vec![Entry::<Word>{internal_count:0,word:Word::default(),block_id:255};num_entries],
            cur:vec![Entry::<Word>{internal_count:0, word:Word::default(), block_id:255};num_entries],
            cur_index:0,
            rand_state:init_silly_rand(),
        }
    }
    fn print<W:Write>(&mut self, w:&mut W, ascii: bool) -> io::Result<()>{
        self.evict(true, 0);
        let mut buf = [0u8;512];
        for entry in self.lru.iter() {
            let count = entry.count();
            if count <= 1 {
                continue;
            }
            let mut cur_index = if ascii {print_encode_ascii(&mut buf, &entry.word)} else {print_encode_hex(&mut buf, &entry.word)};
            buf[cur_index] = ' ' as u8;
            cur_index += 1;
            let mut digit = 1000000000;
            while digit >= 10 {
                if count >= digit {
                    buf[cur_index] = ((count / digit) % 10) as u8 + '0' as u8;
                    cur_index += 1;
                }
                digit /= 10;
            }
            buf[cur_index] = (count % 10) as u8 + '0' as u8;
            cur_index += 1;
            buf[cur_index] = '\n' as u8;
            cur_index += 1;
            try!(w.write_all(&buf[..cur_index]));
        }
        Ok(())
    }
    fn eliminate_duplicates(&mut self) {
        let mut validated_index = 0usize;
        assert!(self.cur_index <= self.cur.len());
        for index in 1..cmp::min(self.cur_index, self.cur.len()) {
            if self.cur[validated_index].word == self.cur[index].word {
                if self.cur[validated_index].block_id != self.cur[index].block_id {
                    let del = self.cur[index].count();
                    self.cur[validated_index].block_id = self.cur[index].block_id;
                    self.cur[validated_index].increment(del);
                }
            } else {
                validated_index += 1;
                let entry = self.cur[index].clone();
                self.cur[validated_index] = entry;
            }
        }
        self.cur_index = validated_index + 1;
    }
    fn evict(&mut self, fin:bool, num_candidates: usize) {
        self.cur[..self.cur_index].sort();
        self.eliminate_duplicates();
        let mut index = 0;
        for entry in self.cur[..self.cur_index].iter() {
            let mut found = false;
            while index < self.lru.len() {
                if self.lru[index].is_random_insert() == false && // just forcibly inserted
                    (self.lru[index].word >= entry.word || self.lru[index].count() == 0) {
                        if self.lru[index].word == entry.word {
                            if self.lru[index].block_id != entry.block_id {
                                self.lru[index].set_nonrandom_flag();
                                self.lru[index].increment(entry.count());
                                self.lru[index].block_id = entry.block_id;
                            }
                            found = true;
                        } else if self.lru[index].count() == 0 {
                            self.lru[index] = entry.clone();
                            found = true;
                        } else {
                            assert_eq!(found, false); // we didn't find it yet
                        }
                        break;
                }
                index += 1;
            }
            if fin == false && !found {
                // randomly insert
                let index = find_candidate(&self.lru, &mut self.rand_state, num_candidates);
                self.lru[index] = Entry::<Word>::new(&entry.word, entry.block_id, true);
            }
        }
        if !fin {
            for entry in self.lru.iter_mut() {
                entry.set_nonrandom_flag();
            }
        }
        self.lru.sort();
        self.cur_index = 0;
    }
    fn add(&mut self, block_id:u8, block:&[u8;4096* 1024], index: usize, num_candidates: usize) {
        {
            let mut dst = &mut self.cur[self.cur_index];
            let len = dst.word.slice().len();
            dst.word.slice_mut().clone_from_slice(&block[index..(index + len)]);
            if ends_with_constant(&dst.word) ||starts_with_constant(&dst.word) {
                return; // lets not bother using this
            }
            dst.reset_count_to_one();
            dst.block_id = block_id;
            self.cur_index += 1;
        }
        if self.cur_index == self.cur.len() {
            self.evict(false, num_candidates);
            assert_eq!(self.cur_index, 0);
        }
    }
}

struct FullLRU((),(),(),(),LRU<Word4>,LRU<Word5>,LRU<Word6>,LRU<Word7>,LRU<Word8>,LRU<Word9>,LRU<Word10>,LRU<Word11>,LRU<Word12>,LRU<Word13>,LRU<Word14>,LRU<Word15>,LRU<Word16>,LRU<Word17>,LRU<Word18>,LRU<Word19>,LRU<Word20>,LRU<Word21>,LRU<Word22>,LRU<Word23>,LRU<Word24>, ());
impl FullLRU {
    fn new(scale:usize) -> Self {
        FullLRU((),(),(),(),
                LRU::<Word4>::new(64 * scale),
                LRU::<Word5>::new(64 * scale),
                LRU::<Word6>::new(64 * scale),
                LRU::<Word7>::new(64 * scale),
                LRU::<Word8>::new(64 * scale),
                LRU::<Word9>::new(64 * scale),
                LRU::<Word10>::new(64 * scale),
                LRU::<Word11>::new(32 * scale),
                LRU::<Word12>::new(32 * scale),
                LRU::<Word13>::new(32 * scale),
                LRU::<Word14>::new(32 * scale),
                LRU::<Word15>::new(16 * scale),
                LRU::<Word16>::new(16 * scale),
                LRU::<Word17>::new(16 * scale),
                LRU::<Word18>::new(16 * scale),
                LRU::<Word19>::new(8 * scale),
                LRU::<Word20>::new(8 * scale),
                LRU::<Word21>::new(8 * scale),
                LRU::<Word22>::new(8 * scale),
                LRU::<Word23>::new(8 * scale),
                LRU::<Word24>::new(8 * scale),
                ())
    }
    pub fn print<W:Write>(&mut self, w: &mut W, ascii: bool) -> io::Result<()>{
        eof(&self.3);
        try!(self.4.print(w, ascii));
        try!(self.5.print(w, ascii));
        try!(self.6.print(w, ascii));
        try!(self.7.print(w, ascii));
        try!(self.8.print(w, ascii));
        try!(self.9.print(w, ascii));
        try!(self.10.print(w, ascii));
        try!(self.11.print(w, ascii));
        try!(self.12.print(w, ascii));
        try!(self.13.print(w, ascii));
        try!(self.14.print(w, ascii));
        try!(self.15.print(w, ascii));
        try!(self.16.print(w, ascii));
        try!(self.17.print(w, ascii));
        try!(self.18.print(w, ascii));
        try!(self.19.print(w, ascii));
        try!(self.20.print(w, ascii));
        try!(self.21.print(w, ascii));
        try!(self.22.print(w, ascii));
        try!(self.23.print(w, ascii));
        try!(self.24.print(w, ascii));
        eof(&self.25);
        Ok(())
    }
}
fn evaluate_biggest<Word:Sliceable+Default+Copy+Ord>(_:&() ,_lru: &LRU<Word>, _:&()) -> usize{
    Word::default().slice().len()
}
fn eof(_:&()){}
fn add_block(block_id: u8, block:&[u8;4096 * 1024], size:usize, lru:&mut FullLRU, rle_max: usize, num_candidates: usize) {
    assert!(size <= 4096 * 1024);
    let max_len = evaluate_biggest(&lru.3, &lru.24, &lru.25);
    assert_eq!(max_len, 24);//known a priori...but the above checks it since EOF comes after lru.24
    if size < max_len {
        return;
    }
    let mut rle = 8;
    let mut rle_char = 0u8;
    for index in 0..(size - max_len) {
        let val = block[index];
        if val == rle_char {
            rle += 1;
            if rle > rle_max {
                continue;
            }
        } else {
            rle = 0;
        }
        rle_char = val;

        lru.4.add(block_id, &block, index, num_candidates);
        lru.5.add(block_id, &block, index, num_candidates);
        lru.6.add(block_id, &block, index, num_candidates);
        lru.7.add(block_id, &block, index, num_candidates);
        lru.8.add(block_id, &block, index, num_candidates);
        lru.9.add(block_id, &block, index, num_candidates);
        lru.10.add(block_id, &block, index, num_candidates);
        lru.11.add(block_id, &block, index, num_candidates);
        lru.12.add(block_id, &block, index, num_candidates);
        lru.13.add(block_id, &block, index, num_candidates);
        lru.14.add(block_id, &block, index, num_candidates);
        lru.15.add(block_id, &block, index, num_candidates);
        lru.16.add(block_id, &block, index, num_candidates);
        lru.17.add(block_id, &block, index, num_candidates);
        lru.18.add(block_id, &block, index, num_candidates);
        lru.19.add(block_id, &block, index, num_candidates);
        lru.20.add(block_id, &block, index, num_candidates);
        lru.21.add(block_id, &block, index, num_candidates);
        lru.22.add(block_id, &block, index, num_candidates);
        lru.23.add(block_id, &block, index, num_candidates);
        lru.24.add(block_id, &block, index, num_candidates);
        eof(&lru.25);
    }
}


fn read_all<R:Read>(mut reader: R,
                      buf: &mut [u8]) -> Result<usize>{
    
    let mut offset = 0;
    loop {
        match reader.read(&mut buf[offset..]) {
            Ok(size) => {
                offset += size;
                if size == 0 {
                    break;
                }
            },
            Err(err) => {
                match err.kind() {
                    ErrorKind::Interrupted => continue,
                    _ => return Err(err),
                }
            },
        }
    }
    Ok(offset)
}

fn main() {
    let _ = writeln!(std::io::stderr(), "Initialization");
    let mut block = [0u8;4096 * 1024];
    let mut block_id = 0;
    let mut rle_max = 4;
    let mut num_candidates = 16;
    let mut scale = 1024usize * 1024usize;
    let mut ascii = false;
    for argument in env::args().skip(1) {
        if argument == "-ascii" {
            ascii = true;
        }
        if argument.starts_with("-s") {
            scale = argument.trim_matches('-').trim_matches('s').parse::<i64>().unwrap() as usize;
        }
        if argument.starts_with("-c") {
            num_candidates = argument.trim_matches('-').trim_matches('c').parse::<i64>().unwrap() as usize;
        }
        if argument.starts_with("-r") {
            rle_max = argument.trim_matches('-').trim_matches('r').parse::<i64>().unwrap() as usize;
        }
    }
    let mut lru = FullLRU::new(scale);
    for argument in env::args().skip(1) {
        if argument == "-ascii" {
            continue;
        }
        if argument.starts_with("-c") {
            continue;
        }
        if argument.starts_with("-s") {
            continue;
        }
        if argument.starts_with("-r") {
            continue;
        }
        let _ = writeln!(std::io::stderr(), "Processing {}", argument);
        let file = File::open(argument).unwrap();
        let _ = std::io::stderr().flush();
        match read_all(&file, &mut block[..]) {
            Ok(size) => add_block(block_id, &block, size, &mut lru, rle_max, num_candidates),
            Err(err) => panic!(err),
        }
        block_id += 1;
    }
    let _ = writeln!(std::io::stderr(), "Finished");
    let _ = std::io::stderr().flush();
    {
        {
            let stdio = io::stdout();
            lru.print(&mut BufWriter::new(stdio.lock()), ascii).unwrap();
        }
        let _ = io::stdout().flush();
    }
}
