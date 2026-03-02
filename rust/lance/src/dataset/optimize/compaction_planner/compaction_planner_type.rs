// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[derive(Debug, PartialEq, Eq, Copy, Hash, Clone, serde::Deserialize, serde::Serialize)]
pub enum CompactionPlannerType {
    Default = 0,
    Bounded = 1,
}

// todo:  各种转换 以及如何暴露python接口 如何暴露java接口