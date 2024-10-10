#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct DataSet<T: Clone> {
    input_data: Vec<T>,
    input_sample_starts_and_lengths: Vec<(usize, usize)>,
    output_data: Vec<T>,
    output_sample_starts_and_lengths: Vec<(usize, usize)>,
}

impl<T: Clone> DataSet<T> {
    #[inline]
    pub fn new(input_data: Vec<Vec<T>>, output_data: Vec<Vec<T>>) -> Self {
        if input_data.len() != output_data.len() { panic!("Input and output data must have same len") };

        let mut data_set: DataSet<T> = DataSet { 
            input_data: Vec::default(), 
            input_sample_starts_and_lengths: Vec::default(),
            output_data: Vec::default(), 
            output_sample_starts_and_lengths: Vec::default(),
        };

        data_set.input_sample_starts_and_lengths = Vec::with_capacity(input_data.len());
        data_set.input_sample_starts_and_lengths.push((0, input_data[0].len()));
        data_set.output_sample_starts_and_lengths = Vec::with_capacity(output_data.len());
        data_set.output_sample_starts_and_lengths.push((0, output_data[0].len()));

        data_set.input_data = [data_set.input_data, input_data[0].clone()].concat();
        data_set.output_data = [data_set.output_data, output_data[0].clone()].concat();

        for s in 0..input_data.len() {
            let prev_input_sample = data_set.input_sample_starts_and_lengths.last().unwrap();
            let prev_output_sample = data_set.output_sample_starts_and_lengths.last().unwrap();
            let input_start = prev_input_sample.0 + prev_input_sample.1;
            let output_start = prev_output_sample.0 + prev_output_sample.1;

            data_set.input_sample_starts_and_lengths.push((input_start, input_data[s].len()));
            data_set.output_sample_starts_and_lengths.push((output_start, output_data[s].len()));

            data_set.input_data = [data_set.input_data, input_data[s].clone()].concat();
            data_set.output_data = [data_set.output_data, output_data[s].clone()].concat();
        }

        data_set
    }

    #[inline]
    pub fn new_combined(input_and_output_data: Vec<(Vec<T>, Vec<T>)>) -> Self {
        let mut data_set: DataSet<T> = DataSet { 
            input_data: Vec::default(), 
            input_sample_starts_and_lengths: Vec::default(),
            output_data: Vec::default(), 
            output_sample_starts_and_lengths: Vec::default(),
        };

        data_set.input_sample_starts_and_lengths = Vec::with_capacity(input_and_output_data.len());
        data_set.input_sample_starts_and_lengths.push((0, input_and_output_data[0].0.len()));
        data_set.output_sample_starts_and_lengths = Vec::with_capacity(input_and_output_data.len());
        data_set.output_sample_starts_and_lengths.push((0, input_and_output_data[0].1.len()));

        data_set.input_data = [data_set.input_data, input_and_output_data[0].0.clone()].concat();
        data_set.output_data = [data_set.output_data, input_and_output_data[0].1.clone()].concat();

        for s in 0..input_and_output_data.len() {
            let prev_input_sample = data_set.input_sample_starts_and_lengths.last().unwrap();
            let prev_output_sample = data_set.output_sample_starts_and_lengths.last().unwrap();
            let input_start = prev_input_sample.0 + prev_input_sample.1;
            let output_start = prev_output_sample.0 + prev_output_sample.1;

            data_set.input_sample_starts_and_lengths.push((input_start, input_and_output_data[s].0.len()));
            data_set.output_sample_starts_and_lengths.push((output_start, input_and_output_data[s].1.len()));

            data_set.input_data = [data_set.input_data, input_and_output_data[s].0.clone()].concat();
            data_set.output_data = [data_set.output_data, input_and_output_data[s].1.clone()].concat();
        }

        data_set
    }
}

// TESTS DESPERATELY NEEDED