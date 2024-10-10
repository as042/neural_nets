#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct DataSet<T: Clone + Default> {
    input_data: Vec<T>,
    input_sample_starts_n_lengths: Vec<(usize, usize)>,
    output_data: Vec<T>,
    output_sample_starts_n_lengths: Vec<(usize, usize)>,
}

impl<T: Clone + Default> DataSet<T> {
    #[inline]
    pub fn new(input_data: Vec<Vec<T>>, output_data: Vec<Vec<T>>) -> Self {
        if input_data.len() != output_data.len() { panic!("Input and output data must have same len") };

        let mut data_set: DataSet<T> = DataSet::default();

        data_set.input_sample_starts_n_lengths = Vec::with_capacity(input_data.len());
        data_set.input_sample_starts_n_lengths.push((0, input_data[0].len()));
        data_set.output_sample_starts_n_lengths = Vec::with_capacity(output_data.len());
        data_set.output_sample_starts_n_lengths.push((0, output_data[0].len()));

        data_set.input_data = [data_set.input_data, input_data[0].clone()].concat();
        data_set.output_data = [data_set.output_data, output_data[0].clone()].concat();

        for s in 0..input_data.len() {
            // let start = data_set.input_sample_starts_n_lengths

            // data_set.input_sample_starts_n_lengths.push((0, input_data[0].len()));
            // data_set.output_sample_starts_n_lengths.push((0, output_data[0].len()));

            // data_set.input_data = [data_set.input_data, input_data[0].clone()].concat();
            // data_set.output_data = [data_set.output_data, output_data[0].clone()].concat();
        }

        data_set
    }
}