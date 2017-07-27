use std::fs::File;
use std::io::{Read,ErrorKind,Result, Write, BufWriter};
use std::io;
use std::env;
use std::cmp;


#[derive(Clone,Copy,PartialEq,Eq)]
struct Word4(pub [u8;4]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word5(pub [u8;5]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word6(pub [u8;6]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word7(pub [u8;7]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word8(pub [u8;8]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word9(pub [u8;9]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word10(pub [u8;10]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word11(pub [u8;11]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word12(pub [u8;12]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word13(pub [u8;13]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word14(pub [u8;14]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word15(pub [u8;15]);
#[derive(Clone,Copy,PartialEq,Eq)]
struct Word16(pub [u8;16]);

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

#[derive(Copy,Clone,PartialEq,Eq)]
struct Entry<Word:Sliceable+Ord+Clone+Copy> {
    count: i32,
    word: Word,
    block_id:u8
}

impl<Word:Sliceable+Ord+Copy> PartialOrd for Entry<Word> {
    fn partial_cmp(&self, other:&Self) -> Option<cmp::Ordering> {
        self.word.partial_cmp(&other.word)
    }
}
impl<Word:Sliceable+Ord+Copy> Ord for Entry<Word> {
    fn cmp(&self, other:&Self) -> cmp::Ordering {
        self.word.cmp(&other.word)
    }
}

fn init_silly_rand() -> u64{
    18446744073709551557u64
}
fn silly_rand(state:&mut u64) -> u64{
    *state = state.wrapping_mul(6364136223846793005u64);
    *state += 1442695040888963407u64;
    return *state;
}
fn find_candidate<Word:Sliceable+Ord+Default+Copy>(lru:&Vec<Entry<Word>>, rand_state:&mut u64) -> usize {
    let mut best_index = silly_rand(rand_state) as usize % lru.len();
    let mut best_count = lru[best_index].count.abs();
    let num_candidates = 4;
    for _ in 0..num_candidates {
        let index = silly_rand(rand_state) as usize % lru.len();
        let count = lru[best_index].count.abs();
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
    rand_state: u64,
}

#[cfg(not(feature="ascii"))]
const NTH:[u8;16] = ['0' as u8,'1' as u8,'2' as u8,'3' as u8,'4' as u8,'5' as u8,
             '6' as u8,'7' as u8,'8' as u8,'9' as u8,'a' as u8,'b' as u8,
             'c' as u8,'d' as u8,'e' as u8,'f' as u8];

#[cfg(not(feature="ascii"))]
fn nibble_to_hex(nib:u8) -> u8 {
    NTH[nib as usize]
}
#[cfg(not(feature="ascii"))]
fn print_encode<Word:Sliceable>(buf:&mut [u8;512], word:&Word) -> usize {
    for (index, item) in word.slice().iter().enumerate() {
        buf[index<<1] = nibble_to_hex(*item>>4);
        buf[(index<<1) + 1] = nibble_to_hex(*item&0xf);
    }
    word.slice().len()<<1
}
#[cfg(feature="ascii")]
fn print_encode<Word:Sliceable>(buf:&mut [u8;512], word:&Word) -> usize {
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

impl<Word:Sliceable+Ord+Default+Copy> LRU<Word> {
    fn new(num_entries: usize) -> Self {
        LRU::<Word>{
            lru:vec![Entry::<Word>{count:0,word:Word::default(),block_id:255};num_entries],
            cur:vec![Entry::<Word>{count:0, word:Word::default(), block_id:255};num_entries],
            cur_index:0,
            rand_state:init_silly_rand(),
        }
    }
    fn print<W:Write>(&mut self, w:&mut W) -> io::Result<()>{
        self.evict(true);
        let mut buf = [0u8;512];
        for entry in self.lru.iter() {
            if entry.count == 0 {
                continue;
            }
            let mut cur_index = print_encode(&mut buf, &entry.word);
            buf[cur_index] = ' ' as u8;
            cur_index += 1;
            let count = entry.count;
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
    fn evict(&mut self, fin:bool) {
        self.cur[..self.cur_index].sort();
        let mut index = 0;
        for entry in self.cur[..self.cur_index].iter() {
            let mut found = false;
            while index < self.lru.len() {
                if self.lru[index].count != -1 && // just forcibly inserted
                    (self.lru[index].word >= entry.word || self.lru[index].count == 0) {
                        if self.lru[index].word == entry.word {
                            if self.lru[index].block_id != entry.block_id {
                                self.lru[index].count += 1;
                                self.lru[index].block_id = entry.block_id;
                            }
                            found = true;
                        } else if self.lru[index].count == 0{
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
                let index = find_candidate(&self.lru, &mut self.rand_state);
                self.lru[index] = Entry::<Word>{
                    count:-1,
                    word:entry.word,
                    block_id:entry.block_id,
                }
            }
        }
        if !fin {
            for entry in self.lru.iter_mut() {
                if entry.count == -1 {
                    entry.count = 1;
                }
            }
        }
        self.lru.sort();
        self.cur_index = 0;
    }
    fn add(&mut self, block_id:u8, block:&[u8;4096* 1024], index: usize) {
        {
            let mut dst = &mut self.cur[self.cur_index];
            let len = dst.word.slice().len();
            dst.word.slice_mut().clone_from_slice(&block[index..(index + len)]);
            if ends_with_constant(&dst.word) ||starts_with_constant(&dst.word) {
                return; // lets not bother using this
            }
            dst.count = 1;
            dst.block_id = block_id;
            self.cur_index += 1;
        }
        if self.cur_index == self.cur.len() {
            self.evict(false);
            assert_eq!(self.cur_index, 0);
        }
    }
}

struct FullLRU((),(),(),(),LRU<Word4>,LRU<Word5>,LRU<Word6>,LRU<Word7>,LRU<Word8>,LRU<Word9>,LRU<Word10>,LRU<Word11>,LRU<Word12>,LRU<Word13>,LRU<Word14>,LRU<Word15>,LRU<Word16>, ());
impl Default for FullLRU {
    fn default() -> Self {
        FullLRU((),(),(),(),
                LRU::<Word4>::new(16 * 1024 * 1024),
                LRU::<Word5>::new(16 * 1024 * 1024),
                LRU::<Word6>::new(16 * 1024 * 1024),
                LRU::<Word7>::new(16 * 1024 * 1024),
                LRU::<Word8>::new(16 * 1024 * 1024),
                LRU::<Word9>::new(8 * 1024 * 1024),
                LRU::<Word10>::new(8 * 1024 * 1024),
                LRU::<Word11>::new(8 * 1024 * 1024),
                LRU::<Word12>::new(8 * 1024 * 1024),
                LRU::<Word13>::new(4 * 1024 * 1024),
                LRU::<Word14>::new(4 * 1024 * 1024),
                LRU::<Word15>::new(4 * 1024 * 1024),
                LRU::<Word16>::new(4 * 1024 * 1024),
                ())
    }
}
impl FullLRU {
    pub fn print<W:Write>(&mut self, w: &mut W) -> io::Result<()>{
        eof(&self.3);
        try!(self.4.print(w));
        try!(self.5.print(w));
        try!(self.6.print(w));
        try!(self.7.print(w));
        try!(self.8.print(w));
        try!(self.9.print(w));
        try!(self.10.print(w));
        try!(self.11.print(w));
        try!(self.12.print(w));
        try!(self.13.print(w));
        try!(self.14.print(w));
        try!(self.15.print(w));
        try!(self.16.print(w));
        eof(&self.17);
        Ok(())
    }
}
fn evaluate_biggest<Word:Sliceable+Default+Copy+Ord>(_:&() ,_lru: &LRU<Word>, _:&()) -> usize{
    Word::default().slice().len()
}
fn eof(_:&()){}
fn add_block(block_id: u8, block:&[u8;4096 * 1024], size:usize, lru:&mut FullLRU, rle_max: usize) {
    assert!(size <= 4096 * 1024);
    let max_len = evaluate_biggest(&lru.3, &lru.16, &lru.17);
    assert_eq!(max_len, 16);//known a priori...but the above checks it since EOF comes after lru.16
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

        lru.4.add(block_id, &block, index);
        lru.5.add(block_id, &block, index);
        lru.6.add(block_id, &block, index);
        lru.7.add(block_id, &block, index);
        lru.8.add(block_id, &block, index);
        lru.9.add(block_id, &block, index);
        lru.10.add(block_id, &block, index);
        lru.11.add(block_id, &block, index);
        lru.12.add(block_id, &block, index);
        lru.13.add(block_id, &block, index);
        lru.14.add(block_id, &block, index);
        lru.15.add(block_id, &block, index);
        lru.16.add(block_id, &block, index);
        eof(&lru.17);
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
    let mut block = [0u8;4096 * 1024];
    let mut lru = FullLRU::default();
    let mut block_id = 0;
    for argument in env::args().skip(1) {
        let file = File::open(argument).unwrap();
        match read_all(&file, &mut block[..]) {
            Ok(size) => add_block(block_id, &block, size, &mut lru, 4),
            Err(err) => panic!(err),
        }
        block_id += 1;
    }
    {
        let stdio = io::stdout();
        lru.print(&mut BufWriter::new(stdio.lock())).unwrap();
    }
}
