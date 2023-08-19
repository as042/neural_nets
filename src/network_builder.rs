#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct NetworkBuilder {
    input_neurons: usize,
}

impl NetworkBuilder {
    /// Creates a new `NetworkBuilder`.
    pub(crate) fn new() -> Self {
        NetworkBuilder::default()
    }

    pub fn add_input_layer() {

    }
}