fn main() {
    let a = 0;
    let b = 1;
    let mut c = 2;
    let mut d = 3;

    a + b;
    &a + b;
    a + &b;
    &a + &b;
    c + d;
    &d + c;
    // (&mut d) += c; // error

}