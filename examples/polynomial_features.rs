//! Expanding features into polynomial and interaction terms.

use machlearn::{PolynomialFeatures, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![[2.0, 3.0], [1.0, 5.0]];

    let fitted = PolynomialFeatures::new(2)?.fit(records.view())?;
    println!("combinations: {:?}", fitted.combinations());
    println!("expanded:\n{:?}", fitted.transform(records.view())?);

    let interactions = PolynomialFeatures::new(2)?
        .with_include_bias(false)
        .with_interaction_only(true)
        .fit(records.view())?;
    println!(
        "interaction-only combinations: {:?}",
        interactions.combinations()
    );
    println!(
        "interactions:\n{:?}",
        interactions.transform(records.view())?
    );
    Ok(())
}
