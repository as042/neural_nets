use super::activation_fn::ActivationFn;
use super::layer::Layer;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
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
    pub fn builder() -> LayoutBuilder {
        LayoutBuilder::new()
    }

    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct LayoutBuilder {
    layers: Vec<Layer>,
}

impl LayoutBuilder {
    #[inline]
    pub fn new() -> Self {
        LayoutBuilder::default()
    }

    #[inline]
    pub fn input_layer(mut self, num_neurons: usize) -> Self {
        self.layers.push(Layer::input(num_neurons));
        self
    }

    #[inline]
    pub fn feed_forward_layer(mut self, activation_fn: ActivationFn, num_neurons: usize) -> Self {
        self.layers.push(Layer::feed_forward(num_neurons, activation_fn));
        self
    }

    #[inline]
    pub fn build(self) -> Layout {
        Layout { layers: self.layers }
    }
}