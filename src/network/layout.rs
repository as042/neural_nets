use super::layer::Layer;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd, Hash)]
pub struct Layout {
    layers: Vec<Layer>,
}

impl Layout {
    #[inline]
    pub fn new(layers: &Vec<Layer>) -> Self {
        Layout { 
            layers: layers.to_vec(),
        }
    }

    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}