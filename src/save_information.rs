use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::autodiff::real::Real;
use crate::prelude::{Layout, Params};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct SaveInformation {
    file_name: String,
    notation: FileNotation,
}

impl SaveInformation {
    #[inline]
    pub fn new(file_name: impl AsRef<Path>, notation: FileNotation) -> Self {
        SaveInformation { 
            file_name: file_name.as_ref().to_string_lossy().to_string(), 
            notation, 
        }
    }

    #[inline]
    pub fn file_name(&self) -> &String {
        &self.file_name
    }

    #[inline]
    pub fn notation(&self) -> FileNotation {
        self.notation
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub enum FileNotation {
    #[default]
    Bytes,
    JSON,
    RON,
    TOML,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct NetworkSaveData<T: Real + Serialize> {
    pub(crate) layout: Layout,
    pub(crate) params: Params<T>,
}