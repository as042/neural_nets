#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct DataSet<T: Clone> {
    input_data: Vec<T>,
    input_sample_starts_and_lengths: Vec<(usize, usize)>,
    output_data: Vec<T>,
    output_sample_starts_and_lengths: Vec<(usize, usize)>,
}

impl<T: Clone> DataSet<T> {
    #[inline]
    pub fn builder() -> DataSetBuilder<T> {
        DataSetBuilder {
            input_data: Vec::default(),
            output_data: Vec::default(),
        }
    }

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

        for s in 1..input_data.len() {
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

        for s in 1..input_and_output_data.len() {
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

    #[inline]
    pub fn input_data(&self) -> &Vec<T> {
        &self.input_data
    }

    #[inline]
    pub fn input_sample_starts_and_lengths(&self) -> &Vec<(usize, usize)> {
        &self.input_sample_starts_and_lengths
    }

    #[inline]
    pub fn output_data(&self) -> &Vec<T> {
        &self.output_data
    }

    #[inline]
    pub fn output_sample_starts_and_lengths(&self) -> &Vec<(usize, usize)> {
        &self.output_sample_starts_and_lengths
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.input_sample_starts_and_lengths.len() != self.output_sample_starts_and_lengths.len() { panic!("Data set must have same number of inputs and outputs") };

        self.input_sample_starts_and_lengths.len()
    }

    #[inline]
    pub fn nth_input(&self, n: usize) -> &[T] {
        let start = self.input_sample_starts_and_lengths[n].0;
        let len = self.input_sample_starts_and_lengths[n].1;

        &self.input_data[start..start + len]
    }

    #[inline]
    pub fn nth_output(&self, n: usize) -> &[T] {
        let start = self.output_sample_starts_and_lengths[n].0;
        let len = self.output_sample_starts_and_lengths[n].1;

        &self.output_data[start..start + len]
    }

    #[inline]
    pub fn nth_sample(&self, n: usize) -> (&[T], &[T]) {
        (self.nth_input(n), self.nth_output(n))
    }
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct DataSetBuilder<T: Clone> {
    input_data: Vec<Vec<T>>,
    output_data: Vec<Vec<T>>,
}

impl<T: Clone> DataSetBuilder<T> {
    #[inline]
    pub fn input(mut self, input: Vec<T>) -> Self {
        self.input_data.push(input);
        self
    }

    #[inline]
    pub fn output(mut self, output: Vec<T>) -> Self {
        self.output_data.push(output);
        self
    }

    #[inline]
    pub fn sample(mut self, input: Vec<T>, output: Vec<T>) -> Self {
        self = self.input(input);
        self = self.output(output);
        self
    }

    #[inline]
    pub fn build(self) -> DataSet<T> {
        DataSet::new(self.input_data, self.output_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data_set1 = DataSet::new(vec![vec![0.1, 0.3], vec![-0.15, 0.2]], vec![vec![0.5, -0.6], vec![-0.2, -0.34]]);
    
        let data_set2 = DataSet {
            input_data: vec![0.1, 0.3, -0.15, 0.2],
            input_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
            output_data: vec![0.5, -0.6, -0.2, -0.34],
            output_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
        };
    
        assert_eq!(data_set1, data_set2);
    }
    
    #[test]
    fn test_new_combined() {
        let data_set1 = DataSet::new_combined(vec![(vec![0.1, 0.3], vec![0.5, -0.6]), (vec![-0.15, 0.2], vec![-0.2, -0.34])]);
    
        let data_set2 = DataSet {
            input_data: vec![0.1, 0.3, -0.15, 0.2],
            input_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
            output_data: vec![0.5, -0.6, -0.2, -0.34],
            output_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
        };
    
        assert_eq!(data_set1, data_set2);
    }
    
    #[test]
    fn test_nth_input() {
        let data_set = DataSet::new(vec![vec![0.1, 0.3], vec![-0.15, 0.2]], vec![vec![0.5, -0.6], vec![-0.2, -0.34]]);
    
        assert_eq!(data_set.nth_input(1), [-0.15, 0.2]);
    }
    
    #[test]
    fn test_nth_output() {
        let data_set = DataSet::new(vec![vec![0.1, 0.3], vec![-0.15, 0.2]], vec![vec![0.5, -0.6], vec![-0.2, -0.34]]);
    
        assert_eq!(data_set.nth_output(0), [0.5, -0.6]);
    }
    
    #[test]
    fn test_nth_sample() {
        let data_set = DataSet::new(vec![vec![0.1, 0.3], vec![-0.15, 0.2]], vec![vec![0.5, -0.6], vec![-0.2, -0.34]]);
    
        assert_eq!(data_set.nth_sample(1), ([-0.15, 0.2].as_slice(), [-0.2, -0.34].as_slice()));
    }
    
    #[test]
    fn data_set_builder_test() {
        let data_set1 = DataSet::builder()
            .input(vec![0.1, 0.3])
            .output(vec![0.5, -0.6])
            .sample(vec![-0.15, 0.2], vec![-0.2, -0.34])
            .build();
    
        let data_set2 = DataSet {
            input_data: vec![0.1, 0.3, -0.15, 0.2],
            input_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
            output_data: vec![0.5, -0.6, -0.2, -0.34],
            output_sample_starts_and_lengths: vec![(0, 2), (2, 2)],
        };
    
        assert_eq!(data_set1, data_set2);
    }
}